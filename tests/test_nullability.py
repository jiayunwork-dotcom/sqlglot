import unittest

import duckdb

import sqlglot
from sqlglot.errors import OptimizeError, SchemaError
from sqlglot.optimizer.nullability import infer_nullability

SCHEMA = {
    "orders": {"id": "INT", "status": "VARCHAR", "amount": "INT", "city_id": "INT"},
    "customers": {"id": "INT", "name": "VARCHAR", "city_id": "INT"},
    "cities": {"id": "INT", "name": "VARCHAR"},
}

NOT_NULL = {
    "orders": {"id"},
    "customers": {"id"},
    "cities": {"id", "name"},
}

DIALECT = "duckdb"


class TestNullability(unittest.TestCase):
    maxDiff = None

    def infer(self, sql, not_null=NOT_NULL, schema=SCHEMA):
        return infer_nullability(
            sql, schema=schema, not_null=not_null, dialect=DIALECT
        )

    def assert_nullable(self, expected, sql, **kwargs):
        self.assertEqual(expected, self.infer(sql, **kwargs))

    def assert_names(self, expected_names, sql):
        result = self.infer(sql)
        self.assertEqual(expected_names, [name for name, _ in result])

    # ------------------------------------------------------------------
    # Basic propagation: aliases, CTEs, derived tables, subqueries, stars
    # ------------------------------------------------------------------

    def test_base_table_columns(self):
        self.assert_nullable(
            [("id", False), ("status", True), ("amount", True)],
            "SELECT id, status, amount FROM orders",
        )

    def test_star_expansion(self):
        self.assert_nullable(
            [("id", False), ("status", True), ("amount", True), ("city_id", True)],
            "SELECT * FROM orders",
        )

    def test_aliases_preserve_nullability(self):
        self.assert_nullable(
            [("oid", False), ("amt", True)],
            "SELECT id AS oid, amount AS amt FROM orders",
        )

    def test_cte_propagation(self):
        sql = """
            WITH c AS (
                SELECT id AS cid, status, amount FROM orders
            )
            SELECT cid, status FROM c
        """
        self.assert_nullable([("cid", False), ("status", True)], sql)

    def test_derived_table_propagation(self):
        sql = """
            SELECT d.oid, d.status
            FROM (SELECT id AS oid, status FROM orders) AS d
        """
        self.assert_nullable([("oid", False), ("status", True)], sql)

    def test_multi_level_subqueries_and_star(self):
        sql = """
            SELECT *
            FROM (
                SELECT * FROM (
                    SELECT id, amount FROM orders
                ) AS d1
            ) AS d2
        """
        self.assert_nullable([("id", False), ("amount", True)], sql)

    def test_qualified_table_star_expands_like_qualify(self):
        result = self.infer("SELECT orders.* FROM orders")
        self.assertEqual(
            ["id", "status", "amount", "city_id"], [name for name, _ in result]
        )

    def test_output_names_match_qualify(self):
        qualified = sqlglot.optimizer.qualify.qualify(
            sqlglot.parse_one("SELECT id + 1, count(*) FROM orders", read=DIALECT),
            schema=SCHEMA,
            dialect=DIALECT,
        )
        expected_names = [select.alias_or_name for select in qualified.selects]
        self.assert_names(expected_names, "SELECT id + 1, count(*) FROM orders")

    # ------------------------------------------------------------------
    # Joins
    # ------------------------------------------------------------------

    def test_inner_join_keeps_nullability(self):
        sql = """
            SELECT o.id, c.id AS cid, c.name
            FROM orders o
            JOIN customers c ON o.id = c.id
        """
        self.assert_nullable(
            [("id", False), ("cid", False), ("name", True)], sql
        )

    def test_left_join_nullifies_right_side(self):
        sql = """
            SELECT o.id, c.id AS cid, c.name
            FROM orders o
            LEFT JOIN customers c ON o.id = c.id
        """
        self.assert_nullable(
            [("id", False), ("cid", True), ("name", True)], sql
        )

    def test_right_join_nullifies_left_side(self):
        sql = """
            SELECT o.id, c.id AS cid, c.name
            FROM orders o
            RIGHT JOIN customers c ON o.id = c.id
        """
        self.assert_nullable(
            [("id", True), ("cid", False), ("name", True)], sql
        )

    def test_full_join_nullifies_both_sides(self):
        sql = """
            SELECT o.id, c.id AS cid
            FROM orders o
            FULL JOIN customers c ON o.id = c.id
        """
        self.assert_nullable([("id", True), ("cid", True)], sql)

    def test_left_join_recovered_by_is_not_null(self):
        sql = """
            SELECT c.id AS cid, c.name
            FROM orders o
            LEFT JOIN customers c ON o.id = c.id
            WHERE c.id IS NOT NULL
        """
        self.assert_nullable([("cid", False), ("name", True)], sql)

    def test_left_join_recovered_by_comparison(self):
        sql = """
            SELECT c.id AS cid
            FROM orders o
            LEFT JOIN customers c ON o.id = c.id
            WHERE c.id > 0
        """
        self.assert_nullable([("cid", False)], sql)

    def test_left_join_recovered_by_equality_to_literal(self):
        sql = """
            SELECT c.id AS cid
            FROM orders o
            LEFT JOIN customers c ON o.id = c.id
            WHERE c.id = 1
        """
        self.assert_nullable([("cid", False)], sql)

    def test_left_join_not_recovered_without_filter(self):
        # A WHERE predicate on the *left* side must not recover right columns.
        sql = """
            SELECT c.id AS cid
            FROM orders o
            LEFT JOIN customers c ON o.id = c.id
            WHERE o.id IS NOT NULL
        """
        self.assert_nullable([("cid", True)], sql)

    def test_full_join_partial_recovery(self):
        sql = """
            SELECT o.id, c.id AS cid
            FROM orders o
            FULL JOIN customers c ON o.id = c.id
            WHERE o.id IS NOT NULL AND c.id > 0
        """
        self.assert_nullable([("id", False), ("cid", False)], sql)

    # ------------------------------------------------------------------
    # WHERE / HAVING guarantees
    # ------------------------------------------------------------------

    def test_where_is_not_null_on_nullable_column(self):
        self.assert_nullable(
            [("amount", False)],
            "SELECT amount FROM orders WHERE amount IS NOT NULL",
        )

    def test_where_equality_to_constant(self):
        self.assert_nullable(
            [("status", False)],
            "SELECT status FROM orders WHERE status = 'paid'",
        )

    def test_where_greater_than_constant(self):
        self.assert_nullable(
            [("amount", False)],
            "SELECT amount FROM orders WHERE amount > 0",
        )

    def test_where_and_combines_guarantees(self):
        sql = "SELECT amount, status FROM orders WHERE amount > 0 AND status = 'paid'"
        self.assert_nullable([("amount", False), ("status", False)], sql)

    def test_where_or_requires_both_branches(self):
        sql = "SELECT amount FROM orders WHERE amount > 0 OR id > 0"
        # amount is not guaranteed in the first branch's sibling rows
        self.assert_nullable([("amount", True)], sql)
        sql2 = "SELECT amount FROM orders WHERE amount > 0 OR amount = 1"
        self.assert_nullable([("amount", False)], sql2)

    def test_having_guarantee(self):
        sql = "SELECT amount FROM orders GROUP BY amount HAVING amount > 0"
        self.assert_nullable([("amount", False)], sql)

    def test_double_negated_is_not_null(self):
        self.assert_nullable(
            [("amount", False)],
            "SELECT amount FROM orders WHERE NOT (NOT (amount IS NOT NULL))",
        )

    def test_double_negated_is_null_guarantees_nothing(self):
        # NOT (NOT (x IS NULL)) means "x IS NULL": retained rows have x NULL,
        # so it must never be treated as a non-NULL guarantee.
        self.assert_nullable(
            [("amount", True)],
            "SELECT amount FROM orders WHERE NOT (NOT (amount IS NULL))",
        )

    def test_guarantee_holds_through_arithmetic_and_coalesce(self):
        # WHERE x IS NOT NULL must make an arithmetic expression containing x
        # non-NULL even when nested inside COALESCE or unary minus.
        self.assert_nullable(
            [("_col_0", False)],
            "SELECT -COALESCE(id - amount, id) FROM orders WHERE amount IS NOT NULL",
        )

    def test_in_and_between_reject_nulls(self):
        self.assert_nullable(
            [("amount", False)],
            "SELECT amount FROM orders WHERE amount IN (1, 2, 3)",
        )
        self.assert_nullable(
            [("amount", False)],
            "SELECT amount FROM orders WHERE amount BETWEEN 1 AND 3",
        )

    def test_bare_boolean_column_guarantee(self):
        schema = {"t": {"flag": "BOOLEAN", "id": "INT"}}
        self.assert_nullable(
            [("flag", False)],
            "SELECT flag FROM t WHERE flag",
            schema=schema,
            not_null={"t": {"id"}},
        )

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def test_count_always_non_null(self):
        sql = "SELECT count(*), count(amount), count(DISTINCT amount) FROM orders"
        self.assert_nullable(
            [("_col_0", False), ("_col_1", False), ("_col_2", False)], sql
        )

    def test_scalar_aggregates_nullable_on_empty_table(self):
        sql = """
            SELECT sum(amount), min(amount), max(amount), avg(amount)
            FROM orders
        """
        self.assert_nullable(
            [
                ("_col_0", True),
                ("_col_1", True),
                ("_col_2", True),
                ("_col_3", True),
            ],
            sql,
        )

    def test_grouped_aggregates_of_not_null_column(self):
        sql = """
            SELECT status, sum(id), min(id), max(id), avg(id)
            FROM orders
            GROUP BY status
        """
        self.assert_nullable(
            [
                ("status", True),
                ("_col_1", False),
                ("_col_2", False),
                ("_col_3", False),
                ("_col_4", False),
            ],
            sql,
        )

    def test_grouped_aggregate_of_nullable_column_stays_nullable(self):
        sql = "SELECT status, sum(amount) FROM orders GROUP BY status"
        self.assert_nullable([("status", True), ("_col_1", True)], sql)

    def test_count_filter_is_non_null(self):
        sql = """
            SELECT count(*) FILTER (WHERE amount > 0)
            FROM orders
            GROUP BY status
        """
        self.assert_nullable([("_col_0", False)], sql)

    def test_rollup_nullifies_grouping_columns(self):
        sql = "SELECT id, count(*) FROM orders GROUP BY ROLLUP(id)"
        self.assert_nullable([("id", True), ("_col_1", False)], sql)

    def test_cube_nullifies_grouping_columns(self):
        sql = "SELECT id, count(*) FROM orders GROUP BY CUBE(id)"
        self.assert_nullable([("id", True), ("_col_1", False)], sql)

    def test_grouping_sets_with_empty_set_nullifies_keys(self):
        sql = "SELECT id, count(*) FROM orders GROUP BY GROUPING SETS ((id), ())"
        self.assert_nullable([("id", True), ("_col_1", False)], sql)

    def test_grouping_sets_without_empty_set_keeps_keys(self):
        sql = "SELECT id, count(*) FROM orders GROUP BY GROUPING SETS ((id))"
        self.assert_nullable([("id", False), ("_col_1", False)], sql)

    def test_grouping_sets_partial_coverage_nullifies_missing_keys(self):
        # id appears only in the first grouping set, so it is NULL-synthesized
        # on rows produced by the second set.
        sql = "SELECT id, amount FROM orders GROUP BY GROUPING SETS ((id), (amount))"
        self.assert_nullable([("id", True), ("amount", True)], sql)

    def test_filter_aggregates_are_nullable_except_count(self):
        sql = """
            SELECT
                count(*) FILTER (WHERE amount > 0),
                sum(id) FILTER (WHERE amount > 0),
                min(id) FILTER (WHERE amount > 0)
            FROM orders
            GROUP BY status
        """
        self.assert_nullable(
            [("_col_0", False), ("_col_1", True), ("_col_2", True)], sql
        )

    def test_rollup_recovered_by_having(self):
        sql = """
            SELECT id, count(*)
            FROM orders
            GROUP BY ROLLUP(id)
            HAVING id IS NOT NULL
        """
        self.assert_nullable([("id", False), ("_col_1", False)], sql)

    # ------------------------------------------------------------------
    # Scalar expressions
    # ------------------------------------------------------------------

    def test_literals_are_non_null(self):
        self.assert_nullable(
            [("one", False), ("s", False), ("n", True)],
            "SELECT 1 AS one, 'x' AS s, NULL AS n",
        )

    def test_coalesce_needs_one_non_null_argument(self):
        sql = "SELECT coalesce(amount, 0), coalesce(amount, status) FROM orders"
        self.assert_nullable([("_col_0", False), ("_col_1", True)], sql)

    def test_case_requires_all_branches_and_else_non_null(self):
        ok = "SELECT CASE WHEN amount > 1 THEN id ELSE 0 END FROM orders"
        self.assert_nullable([("_col_0", False)], ok)

        nullable_then = "SELECT CASE WHEN amount > 1 THEN amount ELSE 0 END FROM orders"
        self.assert_nullable([("_col_0", True)], nullable_then)

        missing_else = "SELECT CASE WHEN amount > 1 THEN id END FROM orders"
        self.assert_nullable([("_col_0", True)], missing_else)

    def test_arithmetic_between_non_null_operands(self):
        sql = "SELECT id + amount, id - id, id * id FROM orders"
        self.assert_nullable(
            [("_col_0", True), ("_col_1", False), ("_col_2", False)], sql
        )

    def test_comparison_between_non_null_operands(self):
        self.assert_nullable(
            [("_col_0", False)], "SELECT id > 1 FROM orders"
        )

    def test_comparison_with_nullable_operand_is_nullable(self):
        self.assert_nullable(
            [("_col_0", True)], "SELECT amount > 1 FROM orders"
        )

    def test_cast_inherits_input_nullability(self):
        self.assert_nullable(
            [("id", False), ("amount", True)],
            "SELECT CAST(id AS VARCHAR) AS id, CAST(amount AS VARCHAR) AS amount FROM orders",
        )

    def test_try_cast_is_nullable(self):
        self.assert_nullable(
            [("status", True)],
            "SELECT TRY_CAST(status AS INTEGER) FROM orders",
        )

    def test_unknown_function_is_nullable(self):
        self.assert_nullable(
            [("_col_0", True)], "SELECT my_unknown_udf(id) FROM orders"
        )

    # ------------------------------------------------------------------
    # Set operations
    # ------------------------------------------------------------------

    def test_union_all_requires_every_arm_non_null(self):
        sql = "SELECT id FROM orders UNION ALL SELECT id FROM customers"
        self.assert_nullable([("id", False)], sql)

        sql2 = "SELECT id FROM orders UNION ALL SELECT amount FROM orders"
        self.assert_nullable([("id", True)], sql2)

    def test_union_distinct_same_rule(self):
        sql = "SELECT id FROM orders UNION SELECT amount FROM orders"
        self.assert_nullable([("id", True)], sql)
        sql2 = "SELECT id FROM orders UNION SELECT id FROM customers"
        self.assert_nullable([("id", False)], sql2)

    def test_intersect_and_except(self):
        self.assert_nullable(
            [("id", False)],
            "SELECT id FROM orders INTERSECT SELECT id FROM customers",
        )
        self.assert_nullable(
            [("id", True)],
            "SELECT id FROM orders EXCEPT SELECT amount FROM orders",
        )

    def test_union_by_name_null_fills_missing_columns(self):
        schema = {
            "t": {"a": "INT", "b": "VARCHAR"},
            "n": {"v": "INT", "b": "VARCHAR"},
        }
        not_null = {"t": {"a"}, "n": {"v"}}
        result = infer_nullability(
            "SELECT a, b FROM t UNION BY NAME SELECT v, 'x' AS b FROM n",
            schema=schema,
            not_null=not_null,
            dialect=DIALECT,
        )
        nullability = dict(result)
        # a and v exist in only one arm, so they are NULL-filled overall.
        self.assertTrue(nullability["a"])
        self.assertTrue(nullability["v"])
        self.assertTrue(nullability["b"])

    # ------------------------------------------------------------------
    # Window functions
    # ------------------------------------------------------------------

    def test_ranking_window_functions_are_non_null(self):
        sql = """
            SELECT
                row_number() OVER (ORDER BY id),
                rank() OVER (ORDER BY id),
                dense_rank() OVER (ORDER BY id)
            FROM orders
        """
        self.assert_nullable(
            [("_col_0", False), ("_col_1", False), ("_col_2", False)], sql
        )

    def test_other_window_functions_are_nullable(self):
        self.assert_nullable(
            [("_col_0", True)],
            "SELECT sum(amount) OVER (PARTITION BY status) FROM orders",
        )

    def test_count_window_is_non_null(self):
        self.assert_nullable(
            [("_col_0", False)], "SELECT count(*) OVER () FROM orders"
        )

    # ------------------------------------------------------------------
    # EXISTS and scalar subqueries
    # ------------------------------------------------------------------

    def test_exists_is_non_null(self):
        sql = """
            SELECT EXISTS (SELECT 1 FROM customers WHERE customers.id = orders.id)
            FROM orders
        """
        self.assert_nullable([("_col_0", False)], sql)

    def test_scalar_subquery_is_nullable(self):
        self.assert_nullable(
            [("_col_0", True)],
            "SELECT (SELECT max(id) FROM customers)",
        )

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_non_query_raises_optimize_error(self):
        for sql in (
            "INSERT INTO orders VALUES (1)",
            "CREATE TABLE x AS SELECT 1",
            "UPDATE orders SET id = 1",
        ):
            with self.subTest(sql=sql):
                with self.assertRaises(OptimizeError):
                    self.infer(sql)

    def test_unknown_table_and_column_raise_optimize_error(self):
        with self.assertRaises(OptimizeError):
            self.infer("SELECT id FROM unknown_table")
        with self.assertRaises(OptimizeError):
            self.infer("SELECT unknown_column FROM orders")

    def test_not_null_unknown_table_raises_schema_error(self):
        with self.assertRaises(SchemaError) as ctx:
            self.infer("SELECT 1", not_null={"unknown_table": {"id"}})
        self.assertIn("unknown_table", str(ctx.exception))

    def test_not_null_unknown_column_raises_schema_error(self):
        with self.assertRaises(SchemaError) as ctx:
            self.infer("SELECT 1", not_null={"orders": {"unknown_column"}})
        self.assertIn("unknown_column", str(ctx.exception))
        self.assertIn("orders", str(ctx.exception))

    def test_db_qualified_not_null_key(self):
        schema = {"db": {"t": {"a": "INT"}}}
        result = infer_nullability(
            "SELECT a FROM db.t",
            schema=schema,
            not_null={"db.t": {"a"}},
            dialect=DIALECT,
        )
        self.assertEqual([("a", False)], result)

    def test_parsed_expression_is_not_mutated(self):
        expression = sqlglot.parse_one(
            "SELECT id FROM orders WHERE id IS NOT NULL", dialect=DIALECT
        )
        before = expression.sql()
        self.infer(expression)
        self.assertEqual(before, expression.sql())

    def test_string_and_parsed_inputs_agree(self):
        sql = "SELECT id FROM orders"
        from_string = self.infer(sql)
        from_expression = self.infer(sqlglot.parse_one(sql, dialect=DIALECT))
        self.assertEqual(from_string, from_expression)

    def test_no_not_null_means_everything_nullable(self):
        result = infer_nullability(
            "SELECT id, status FROM orders", schema=SCHEMA, dialect=DIALECT
        )
        self.assertTrue(all(nullable for _, nullable in result))

    def test_with_query_through_alias_and_star(self):
        sql = """
            WITH c1 AS (
                SELECT id AS cid, amount FROM orders
            ),
            c2 AS (
                SELECT * FROM c1
            )
            SELECT d.cid
            FROM (SELECT cid FROM c2) d
        """
        self.assert_nullable([("cid", False)], sql)

    def test_count_column_form(self):
        self.assert_nullable(
            [("_col_0", False)], "SELECT count(id) FROM orders"
        )

    def test_left_join_recovery_via_non_id_column(self):
        # A NULL-rejecting predicate on any right-side column removes every
        # NULL-padded row, recovering the right source's NOT NULL columns.
        schema = {
            "orders": {"id": "INT"},
            "customers": {"id": "INT", "name": "VARCHAR", "status": "VARCHAR"},
        }
        not_null = {"orders": {"id"}, "customers": {"id"}}
        sql = """
            SELECT c.id AS cid, c.name
            FROM orders o
            LEFT JOIN customers c ON o.id = c.id
            WHERE c.status = 'paid'
        """
        result = infer_nullability(
            sql, schema=schema, not_null=not_null, dialect=DIALECT
        )
        self.assertEqual([("cid", False), ("name", True)], result)

    def test_left_join_recovery_propagates_inward(self):
        # When the outermost left-joined source is proven present by WHERE,
        # its ON equality proves the intermediate left-joined source is
        # present too (but never the other direction).
        schema = {
            "a": {"id": "INT"},
            "b": {"id": "INT"},
            "c": {"id": "INT", "z": "INT"},
        }
        not_null = {"a": {"id"}, "b": {"id"}, "c": {"id"}}
        recovered = """
            SELECT a.id, b.id, c.id
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON b.id = c.id
            WHERE c.z > 0
        """
        self.assert_nullable(
            [("id", False), ("id", False), ("id", False)],
            recovered,
            schema=schema,
            not_null=not_null,
        )

        not_recovered = """
            SELECT a.id, b.id, c.id
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON b.id = c.id
            WHERE a.id IS NOT NULL
        """
        self.assert_nullable(
            [("id", False), ("id", True), ("id", True)],
            not_recovered,
            schema=schema,
            not_null=not_null,
        )

    # ------------------------------------------------------------------
    # False-negative hunting: constructs that must stay conservative
    # ------------------------------------------------------------------

    def test_false_negatives(self):
        cases = [
            # Nullable column used directly
            "SELECT amount FROM orders",
            # Arithmetic / comparison touching a nullable operand
            "SELECT amount + 1 FROM orders",
            "SELECT id = amount FROM orders",
            # COALESCE where every argument is nullable
            "SELECT coalesce(amount, status) FROM orders",
            # Aggregate of a nullable input
            "SELECT sum(amount) FROM orders",
            # Outer-join nullified side without a recovering predicate
            "SELECT c.name FROM orders o LEFT JOIN customers c ON o.id = c.id",
            # Full join columns
            "SELECT o.id FROM orders o FULL JOIN customers c ON o.id = c.id",
            # No-GROUP-BY aggregate over an empty table
            "SELECT max(id) FROM orders",
            # CASE without ELSE
            "SELECT CASE WHEN amount > 1 THEN id END FROM orders",
            # Scalar subquery may return no rows
            "SELECT (SELECT id FROM orders LIMIT 1)",
            # NULL literal
            "SELECT NULL",
            # Grouping-set subtotal NULLs
            "SELECT id FROM orders GROUP BY ROLLUP(id)",
            # NOT NULL column projected from a UNION with a nullable arm
            "SELECT id FROM orders UNION ALL SELECT amount FROM orders",
            # TRY_CAST can yield NULL
            "SELECT TRY_CAST(status AS BIGINT) FROM orders",
            # Table functions are unknown
            "SELECT x FROM UNNEST([1, 2]) AS u(x)",
        ]
        for sql in cases:
            with self.subTest(sql=sql):
                result = self.infer(sql)
                self.assertTrue(
                    all(nullable for _, nullable in result),
                    msg=f"expected all nullable for: {sql} -> {result}",
                )


class TestJoinRecoveryNullability(unittest.TestCase):
    """
    Outer-join presence recovery: NULL-padded rows must never be mistaken for
    present source rows. Expected results are derived from SQL semantics, not
    from the implementation under test, and cross-checked against DuckDB in
    ``TestJoinRecoveryDuckDB``.
    """

    maxDiff = None

    SCHEMA = {
        "a": {"id": "INT", "v": "INT"},
        "b": {"id": "INT", "v": "INT"},
        "c": {"id": "INT", "v": "INT"},
    }
    NOT_NULL = {"a": {"id"}, "b": {"id"}, "c": {"id"}}

    def infer(self, sql):
        return infer_nullability(
            sql, schema=self.SCHEMA, not_null=self.NOT_NULL, dialect=DIALECT
        )

    def assert_nullable(self, expected, sql):
        self.assertEqual(expected, self.infer(sql), msg=sql)

    # ------------------------------------------------------------------
    # Reported regressions: filtering one side must not recover another
    # side that is still NULL-padded.
    # ------------------------------------------------------------------

    def test_filter_on_b_does_not_recover_independent_c(self):
        # a=(1,NULL), b=(1,NULL), c empty: WHERE keeps the (a,b) match while
        # c stays NULL-padded.
        sql = """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON a.id = c.id
            WHERE b.id IS NOT NULL
        """
        self.assert_nullable([("bid", False), ("cid", True)], sql)

    def test_or_in_on_blocks_recovery_of_partner(self):
        # a=(2,NULL), b empty, c=(9,1): the OR branch matches c while b stays
        # NULL-padded.
        sql = """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON b.id = c.id OR a.id = 2
            WHERE c.id IS NOT NULL
        """
        self.assert_nullable([("bid", True), ("cid", False)], sql)

    def test_coalesce_in_on_blocks_recovery_of_partner(self):
        sql = """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON COALESCE(b.id, 9) = c.id
            WHERE c.id IS NOT NULL
        """
        self.assert_nullable([("bid", True), ("cid", False)], sql)

    def test_null_safe_on_blocks_recovery_of_partner(self):
        # b's join accepts its NULL-padded row, so c matching later cannot
        # prove b exists.
        sql = """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id IS NOT DISTINCT FROM b.id
            LEFT JOIN c ON b.id = c.id
            WHERE c.id IS NOT NULL
        """
        self.assert_nullable([("bid", True), ("cid", False)], sql)

    def test_mixed_or_in_later_on_blocks_chain_recovery(self):
        sql = """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id IS NOT DISTINCT FROM b.id
            LEFT JOIN c ON b.id = c.id OR a.id = c.id
            WHERE c.id IS NOT NULL
        """
        self.assert_nullable([("bid", True), ("cid", False)], sql)

    # ------------------------------------------------------------------
    # Guarantees that must keep working.
    # ------------------------------------------------------------------

    def test_filter_recovers_both_id_and_filtered_column(self):
        for predicate in ("b.v > 0", "b.v = 1", "b.v IS NOT NULL"):
            with self.subTest(predicate=predicate):
                sql = f"""
                    SELECT b.id AS bid, b.v AS bv
                    FROM a LEFT JOIN b ON a.id = b.id
                    WHERE {predicate}
                """
                self.assert_nullable([("bid", False), ("bv", False)], sql)

    def test_unfiltered_nullable_column_stays_nullable_after_recovery(self):
        # b.id is NOT NULL and recovered; b.v has no NOT NULL constraint and is
        # not itself filtered, so it remains nullable.
        sql = """
            SELECT b.id AS bid, b.v AS bv
            FROM a LEFT JOIN b ON a.id = b.id
            WHERE b.id IS NOT NULL
        """
        self.assert_nullable([("bid", False), ("bv", True)], sql)

    def test_pure_equality_chain_recovers_all_not_null_columns(self):
        sql = """
            SELECT a.id AS aid, b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON b.id = c.id
            WHERE c.v > 0
        """
        self.assert_nullable(
            [("aid", False), ("bid", False), ("cid", False)], sql
        )

    def test_chain_extended_with_and_equalities_recovers_all(self):
        sql = """
            SELECT a.id AS aid, b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id AND a.v = b.v
            LEFT JOIN c ON b.id = c.id AND b.v = c.v
            WHERE c.id IS NOT NULL
        """
        self.assert_nullable(
            [("aid", False), ("bid", False), ("cid", False)], sql
        )

    def test_filter_on_b_leaves_unmatched_c_nullable(self):
        sql = """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON b.id = c.id
            WHERE b.id IS NOT NULL
        """
        self.assert_nullable([("bid", False), ("cid", True)], sql)

    def test_inner_and_cross_keep_intrinsic_nullability(self):
        self.assert_nullable(
            [("aid", False), ("bid", False)],
            "SELECT a.id AS aid, b.id AS bid FROM a INNER JOIN b ON a.id = b.id",
        )
        self.assert_nullable(
            [("aid", False), ("bid", False)],
            "SELECT a.id AS aid, b.id AS bid FROM a CROSS JOIN b",
        )

    def test_full_join_never_propagates_presence(self):
        self.assert_nullable(
            [("aid", False), ("bid", True)],
            """
            SELECT a.id AS aid, b.id AS bid
            FROM a FULL JOIN b ON a.id = b.id
            WHERE a.id IS NOT NULL
            """,
        )
        self.assert_nullable(
            [("aid", True), ("bid", False)],
            """
            SELECT a.id AS aid, b.id AS bid
            FROM a FULL JOIN b ON a.id = b.id
            WHERE b.id IS NOT NULL
            """,
        )

    def test_right_join_recovery_through_equality(self):
        sql = """
            SELECT a.id AS aid, b.id AS bid
            FROM a RIGHT JOIN b ON a.id = b.id
            WHERE a.id IS NOT NULL
        """
        self.assert_nullable([("aid", False), ("bid", False)], sql)

    def test_right_join_with_weak_on_is_conservative(self):
        sql = """
            SELECT a.id AS aid, b.id AS bid
            FROM a RIGHT JOIN b ON a.id IS NOT DISTINCT FROM b.id
            WHERE b.id IS NOT NULL
        """
        self.assert_nullable([("aid", True), ("bid", False)], sql)

    def test_or_requires_presence_in_every_branch(self):
        # Only one branch proves c present: c must stay nullable.
        sql = """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON b.id = c.id
            WHERE b.id IS NOT NULL OR c.id IS NOT NULL
        """
        self.assert_nullable([("bid", True), ("cid", True)], sql)

        # Both branches prove b present: b is recovered, c is not mentioned.
        sql = """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON b.id = c.id
            WHERE b.id > 0 OR b.id = 1
        """
        self.assert_nullable([("bid", False), ("cid", True)], sql)

    def test_coalesce_predicate_does_not_seed_presence(self):
        # COALESCE stays non-NULL on a NULL-padded b row, so b is not present;
        # c is directly guaranteed by its own plain-column predicate.
        sql = """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON COALESCE(a.id, 0) = b.id
            LEFT JOIN c ON b.id = c.id
            WHERE c.id IS NOT NULL
        """
        self.assert_nullable([("bid", True), ("cid", False)], sql)

    def test_nested_column_in_where_expression_does_not_seed_presence(self):
        # The column is buried in COALESCE / arithmetic. The predicate is
        # NULL-rejecting for the whole expression, but the source row can
        # still be NULL-padded, so b must stay nullable. c is a direct operand
        # and remains guaranteed.
        self.assert_nullable(
            [("bid", True), ("cid", False)],
            """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON a.id = c.id OR 1 = 1
            WHERE COALESCE(b.id, 0) = c.id
            """,
        )
        self.assert_nullable(
            [("bid", True), ("cid", False)],
            """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON a.id = c.id OR 1 = 1
            WHERE b.id + 1 = c.id
            """,
        )
        self.assert_nullable(
            [("bid", True), ("cid", True)],
            """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON a.id = c.id OR 1 = 1
            WHERE NOT (COALESCE(b.id, 0) IS NULL)
            """,
        )
        # A parenthesized bare column is still a direct operand.
        self.assert_nullable(
            [("bid", False), ("cid", True)],
            """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON b.id = c.id
            WHERE (b.id) IS NOT NULL
            """,
        )

    def test_non_equality_on_clause_blocks_on_recovery(self):
        # b's ON mixes the equality with a range predicate, so it produces no
        # presence edge: c being present cannot recover b through the join.
        sql = """
            SELECT a.id AS aid, b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id AND b.v > 0
            LEFT JOIN c ON b.id = c.id
            WHERE c.id IS NOT NULL
        """
        self.assert_nullable(
            [("aid", False), ("bid", True), ("cid", False)], sql
        )

        # The same chain with a pure-equality ON does recover b.
        pure_sql = """
            SELECT a.id AS aid, b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON b.id = c.id
            WHERE c.id IS NOT NULL
        """
        self.assert_nullable(
            [("aid", False), ("bid", False), ("cid", False)], pure_sql
        )

    # ------------------------------------------------------------------
    # Equivalent rewrites, scoping, aliasing
    # ------------------------------------------------------------------

    def test_derived_table_wrapper_preserves_result(self):
        wrapped = """
            SELECT bid, cid FROM (
                SELECT b.id AS bid, c.id AS cid
                FROM a
                LEFT JOIN b ON a.id = b.id
                LEFT JOIN c ON a.id = c.id
                WHERE b.id IS NOT NULL
            ) d
        """
        direct = """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON a.id = c.id
            WHERE b.id IS NOT NULL
        """
        self.assertEqual(self.infer(wrapped), self.infer(direct))

    def test_forwarding_cte_preserves_result(self):
        sql = """
            WITH d AS (
                SELECT b.id AS bid, c.id AS cid
                FROM a
                LEFT JOIN b ON a.id = b.id
                LEFT JOIN c ON b.id = c.id OR a.id = 2
                WHERE c.id IS NOT NULL
            )
            SELECT bid, cid FROM d
        """
        self.assert_nullable([("bid", True), ("cid", False)], sql)

    def test_alias_shadowing_does_not_leak_between_scopes(self):
        # Both scopes call a source "b"; the inner filter on b must not recover
        # the outer b. Output positions are compared explicitly.
        sql = """
            SELECT b.id AS outer_bid, d.x AS inner_bid
            FROM a
            LEFT JOIN b ON a.id = b.id
            JOIN (
                SELECT b2.id AS x
                FROM a AS a2
                LEFT JOIN b AS b2 ON a2.id = b2.id
                WHERE b2.id IS NOT NULL
            ) d ON d.x = a.id
        """
        # Outer b is still nullable; the inner derived column is not nullable.
        self.assert_nullable([("outer_bid", True), ("inner_bid", False)], sql)

    def test_renaming_table_aliases_is_positionally_stable(self):
        sql1 = """
            SELECT x.id AS xid, y.id AS yid
            FROM a AS x LEFT JOIN b AS y ON x.id = y.id
            WHERE y.id IS NOT NULL
        """
        sql2 = """
            SELECT p.id AS xid, q.id AS yid
            FROM a AS p LEFT JOIN b AS q ON p.id = q.id
            WHERE q.id IS NOT NULL
        """
        self.assertEqual(self.infer(sql1), self.infer(sql2))

    def test_independent_left_joins_order_invariant(self):
        # The two ON clauses reference only the preserved side and their own
        # right table, so swapping b and c must not change positional results
        # after aligning output positions.
        sql1 = """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON a.id = c.id
            WHERE b.id IS NOT NULL
        """
        sql2 = """
            SELECT c.id AS cid, b.id AS bid
            FROM a
            LEFT JOIN c ON a.id = c.id
            LEFT JOIN b ON a.id = b.id
            WHERE b.id IS NOT NULL
        """
        self.assertEqual(
            {name: nullable for name, nullable in self.infer(sql1)},
            {name: nullable for name, nullable in self.infer(sql2)},
        )

    # ------------------------------------------------------------------
    # Input isolation
    # ------------------------------------------------------------------

    def test_inputs_are_not_mutated(self):
        import sqlglot

        sql = """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON b.id = c.id OR a.id = 2
            WHERE c.id IS NOT NULL
        """
        expression = sqlglot.parse_one(sql, dialect=DIALECT)
        schema = {name: dict(columns) for name, columns in self.SCHEMA.items()}
        not_null = {name: set(columns) for name, columns in self.NOT_NULL.items()}
        before_sql = expression.sql()
        before_schema = {name: dict(columns) for name, columns in schema.items()}
        before_nn = {name: set(columns) for name, columns in not_null.items()}

        infer_nullability(
            expression,
            schema=schema,
            not_null=not_null,
            dialect=DIALECT,
        )

        self.assertEqual(before_sql, expression.sql())
        self.assertEqual(before_schema, schema)
        self.assertEqual(before_nn, not_null)

    # ------------------------------------------------------------------
    # Scale: no exponential work on wide joins or deep nesting.
    # ------------------------------------------------------------------

    def test_scale_boundaries_finish_quickly(self):
        import time

        # 30-table pure-equality LEFT JOIN chain.
        width = 30
        tables = {f"t{i}": {"id": "INT", "v": "INT"} for i in range(width)}
        not_null = {f"t{i}": {"id"} for i in range(width)}
        chain_sql = (
            "SELECT "
            + ", ".join(f"t{i}.id" for i in range(width))
            + " FROM t0 "
            + " ".join(
                f"LEFT JOIN t{i} ON t{i - 1}.id = t{i}.id" for i in range(1, width)
            )
            + " WHERE t29.id IS NOT NULL"
        )
        started = time.perf_counter()
        result = infer_nullability(
            chain_sql, schema=tables, not_null=not_null, dialect=DIALECT
        )
        self.assertLess(time.perf_counter() - started, 2.0)
        self.assertTrue(all(not nullable for _, nullable in result))

        # 50 levels of column-forwarding derived tables.
        nested_sql = (
            "SELECT id FROM "
            + " ".join("(SELECT id FROM" for _ in range(50))
            + " t0"
            + ")" * 50
        )
        started = time.perf_counter()
        self.assertEqual(
            [("id", False)],
            infer_nullability(
                nested_sql,
                schema={"t0": {"id": "INT"}},
                not_null={"t0": {"id"}},
                dialect=DIALECT,
            ),
        )
        self.assertLess(time.perf_counter() - started, 2.0)

        # 40 levels of alternating AND/OR, both in WHERE and in an ON clause.
        predicate = "t0.id IS NOT NULL"
        for index in range(40):
            connector = "AND" if index % 2 == 0 else "OR"
            predicate = f"({predicate} {connector} t0.id > {index})"
        where_sql = f"SELECT id FROM t0 WHERE {predicate}"
        started = time.perf_counter()
        infer_nullability(
            where_sql,
            schema={"t0": {"id": "INT"}},
            not_null={"t0": {"id"}},
            dialect=DIALECT,
        )
        self.assertLess(time.perf_counter() - started, 2.0)

        on_predicate = "t0.id = t1.id"
        for index in range(40):
            connector = "AND" if index % 2 == 0 else "OR"
            on_predicate = f"({on_predicate} {connector} t1.id > {index})"
        on_sql = (
            f"SELECT t1.id FROM t0 LEFT JOIN t1 ON {on_predicate} "
            "WHERE t1.id IS NOT NULL"
        )
        started = time.perf_counter()
        infer_nullability(
            on_sql,
            schema={"t0": {"id": "INT"}, "t1": {"id": "INT"}},
            not_null={"t0": {"id"}, "t1": {"id"}},
            dialect=DIALECT,
        )
        self.assertLess(time.perf_counter() - started, 2.0)


class TestNullabilityDuckDBFuzz(unittest.TestCase):
    """
    Cross-checks every inference against actual DuckDB execution using random
    data that respects the declared NOT NULL constraints. A column inferred
    non-NULL must never contain a NULL in any executed result.
    """

    NUM_TRIALS = 20
    NUM_ROWS = 50
    SEED = 1234

    # Mirrors the physical DuckDB tables, including the empty copies used to
    # exercise scalar-aggregate behavior on empty inputs.
    FUZZ_SCHEMA = {
        **SCHEMA,
        "empty_orders": {
            "id": "INT",
            "status": "VARCHAR",
            "amount": "INT",
            "city_id": "INT",
        },
        "empty_customers": {"id": "INT", "name": "VARCHAR", "city_id": "INT"},
    }
    FUZZ_NOT_NULL = {
        **NOT_NULL,
        "empty_orders": {"id"},
        "empty_customers": {"id"},
    }

    def setUp(self):
        self.conn = duckdb.connect()
        rng = __import__("random").Random(self.SEED)
        self.rng = rng

        self.conn.execute(
            "CREATE TABLE orders ("
            "id INTEGER NOT NULL, status VARCHAR, amount INTEGER, city_id INTEGER)"
        )
        self.conn.execute(
            "CREATE TABLE customers ("
            "id INTEGER NOT NULL, name VARCHAR, city_id INTEGER)"
        )
        self.conn.execute(
            "CREATE TABLE cities (id INTEGER NOT NULL, name VARCHAR NOT NULL)"
        )

        orders_rows = []
        for i in range(self.NUM_ROWS):
            orders_rows.append(
                (
                    i + 1,
                    None if rng.random() < 0.4 else rng.choice(["paid", "open", "lost"]),
                    None if rng.random() < 0.4 else rng.randint(-10, 100),
                    None if rng.random() < 0.3 else rng.randint(1, 5),
                )
            )
        self.conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders_rows)

        customers_rows = []
        for i in range(self.NUM_ROWS):
            customers_rows.append(
                (
                    i + 1,
                    None if rng.random() < 0.4 else f"name-{i}",
                    None if rng.random() < 0.3 else rng.randint(1, 5),
                )
            )
        self.conn.executemany(
            "INSERT INTO customers VALUES (?, ?, ?)", customers_rows
        )

        cities_rows = [(i, f"city-{i}") for i in range(1, 6)]
        self.conn.executemany("INSERT INTO cities VALUES (?, ?)", cities_rows)

        # Empty tables to exercise scalar-aggregate NULL behavior.
        self.conn.execute(
            "CREATE TABLE empty_orders AS SELECT * FROM orders WHERE 1 = 0"
        )
        self.conn.execute(
            "CREATE TABLE empty_customers AS SELECT * FROM customers WHERE 1 = 0"
        )

    def tearDown(self):
        self.conn.close()

    def _check_query(self, sql, not_null=None, schema=None):
        inference = infer_nullability(
            sql,
            schema=schema or self.FUZZ_SCHEMA,
            not_null=self.FUZZ_NOT_NULL if not_null is None else not_null,
            dialect=DIALECT,
        )
        executed = self.conn.execute(sql).fetchdf()
        # Compare positionally: engine-generated column names differ from
        # sqlglot's synthetic names for unaliased expressions.
        self.assertEqual(
            len(inference),
            len(executed.columns),
            msg=f"column count mismatch for: {sql}",
        )
        for index, ((name, nullable), series) in enumerate(
            zip(inference, executed.columns)
        ):
            if not nullable:
                self.assertFalse(
                    bool(executed[series].isna().any()),
                    msg=(
                        f"false non-NULL inference for column {index} ('{name}') "
                        f"in: {sql}\ninference={inference}"
                    ),
                )

    def test_randomized_projections(self):
        templates = [
            "SELECT {idcol} FROM orders",
            "SELECT {idcol} AS oid, amount FROM orders",
            "SELECT amount FROM orders WHERE amount IS NOT NULL",
            "SELECT amount FROM orders WHERE amount > 0",
            "SELECT status FROM orders WHERE status = 'paid'",
            "SELECT amount FROM orders WHERE amount > 0 OR amount = 1",
            "SELECT coalesce(amount, 0) FROM orders",
            "SELECT coalesce(amount, city_id) FROM orders",
            "SELECT CASE WHEN amount > 1 THEN {idcol} ELSE 0 END FROM orders",
            "SELECT CASE WHEN amount > 1 THEN {idcol} END FROM orders",
            "SELECT {idcol} + {idcol}, {idcol} * {idcol} FROM orders",
            "SELECT amount + 1, {idcol} > 0 FROM orders",
            "SELECT CAST({idcol} AS VARCHAR), TRY_CAST(amount AS VARCHAR) FROM orders",
            "SELECT count(*), count(amount) FROM orders",
            "SELECT sum({idcol}), min({idcol}), max({idcol}), avg({idcol}) FROM orders",
            "SELECT sum(amount), min(amount) FROM empty_orders",
            "SELECT {idcol}, count(*), sum(amount) FROM empty_orders GROUP BY {idcol}",
            "SELECT status, count(*), sum({idcol}) FROM orders GROUP BY status",
            "SELECT {idcol}, count(*) FROM orders GROUP BY ROLLUP({idcol})",
            "SELECT {idcol}, count(*) FROM orders GROUP BY GROUPING SETS (({idcol}), ())",
            "SELECT row_number() OVER (ORDER BY {idcol}), "
            "rank() OVER (), dense_rank() OVER () FROM orders",
            "SELECT sum(amount) OVER (PARTITION BY status) FROM orders",
            "SELECT count(*) OVER () FROM orders",
            "SELECT EXISTS (SELECT 1 FROM customers WHERE customers.id = orders.id) FROM orders",
            "SELECT (SELECT max(id) FROM customers) FROM orders",
            "SELECT o.{idcol}, c.id AS cid, c.name FROM orders o "
            "LEFT JOIN customers c ON o.id = c.id",
            "SELECT c.id AS cid, c.name FROM orders o "
            "LEFT JOIN customers c ON o.id = c.id WHERE c.id IS NOT NULL",
            "SELECT c.id AS cid FROM orders o "
            "LEFT JOIN customers c ON o.id = c.id WHERE c.id > 0",
            "SELECT o.{idcol}, c.id AS cid FROM orders o "
            "RIGHT JOIN customers c ON o.id = c.id",
            "SELECT o.{idcol}, c.id AS cid FROM orders o "
            "FULL JOIN customers c ON o.id = c.id",
            "SELECT o.{idcol}, c.id AS cid FROM orders o JOIN customers c ON o.id = c.id",
            "SELECT {idcol} FROM orders UNION ALL SELECT id FROM customers",
            "SELECT {idcol} FROM orders UNION ALL SELECT amount FROM orders",
            "SELECT {idcol} FROM orders INTERSECT SELECT id FROM customers",
            "SELECT {idcol} FROM orders EXCEPT SELECT amount FROM orders",
            "WITH c AS (SELECT id AS cid, status FROM orders) SELECT cid, status FROM c",
            "SELECT * FROM (SELECT id, amount FROM orders) d",
            "SELECT regexp_full_match(status, 'paid'), least(id, id) FROM orders",
            "SELECT -id, NOT (id > 1) FROM orders",
        ]
        for trial in range(self.NUM_TRIALS):
            for template in templates:
                sql = template.format(idcol="id")
                with self.subTest(trial=trial, sql=sql):
                    self._check_query(sql)

    def test_randomized_filter_combinations(self):
        # Random AND/OR nests of NULL-rejecting predicates must never produce
        # a false non-NULL claim. Predicates are generated per column type.
        numeric_ops = ["IS NOT NULL", "> 0", "= 1", "IS NULL"]
        text_ops = ["IS NOT NULL", "= 'paid'", "IS NULL"]
        columns = {
            "id": numeric_ops,
            "amount": numeric_ops,
            "city_id": numeric_ops,
            "status": text_ops,
        }
        for trial in range(self.NUM_TRIALS):
            predicates = [
                f"{column} {self.rng.choice(ops)}" for column, ops in columns.items()
            ]
            predicate = predicates[0]
            for extra in predicates[1:]:
                connector = self.rng.choice(["AND", "OR"])
                predicate = f"({predicate} {connector} {extra})"
            sql = f"SELECT id, amount, status, city_id FROM orders WHERE {predicate}"
            with self.subTest(trial=trial):
                self._check_query(sql)

    def test_randomized_join_chains(self):
        for trial in range(self.NUM_TRIALS):
            sides = [self.rng.choice(["LEFT", "INNER", "RIGHT", "FULL"]) for _ in range(2)]
            joins = []
            sources = ["o"]
            for index, side in enumerate(sides):
                alias = f"c{index}"
                joins.append(f"{side} JOIN customers {alias} ON o.id = {alias}.id")
                sources.append(alias)
            projections = ", ".join(
                f"{source}.id AS {source}_id" for source in sources
            )
            sql = "SELECT " + projections + " FROM orders o " + " ".join(joins)
            with self.subTest(trial=trial, sql=sql):
                self._check_query(sql)

    def test_randomized_union_trees(self):
        # Each arm projects the same two output names so the outer set
        # operations can align them positionally. Both columns are nullable
        # on at least one arm, keeping the check conservative.
        base_queries = [
            "SELECT id, amount FROM orders",
            "SELECT id, city_id FROM orders",
        ]
        connectors = ["UNION ALL", "UNION", "INTERSECT", "EXCEPT"]
        for trial in range(self.NUM_TRIALS):
            sql = base_queries[trial % len(base_queries)]
            for _ in range(3):
                other = self.rng.choice(base_queries)
                connector = self.rng.choice(connectors)
                sql = f"{sql} {connector} {other}"
            with self.subTest(trial=trial):
                self._check_query(sql)


class TestJoinRecoveryDuckDB(unittest.TestCase):
    """
    Executes the join-recovery queries against real DuckDB tables and checks
    every output position: a column inferred non-NULL (``False``) must never
    contain a NULL in any result row. Data is generated with a fixed seed and
    includes empty tables, unmatched rows, nullable values and duplicate keys,
    all while respecting the declared NOT NULL constraints.

    Uses ``fetchall`` rather than DataFrames so the check has no numpy/pandas
    dependency. Expected inference results asserted elsewhere are hand-derived
    from SQL semantics, never produced by the implementation under test.
    """

    SEED = 20261114
    SCHEMA = {
        "a": {"id": "INT", "v": "INT"},
        "b": {"id": "INT", "v": "INT"},
        "c": {"id": "INT", "v": "INT"},
    }
    NOT_NULL = {"a": {"id"}, "b": {"id"}, "c": {"id"}}

    # (sql, expected inference) for the two reported counterexamples and the
    # COALESCE / null-safe variants. These expectations are hand-written.
    HANDED_EXPECTATIONS = [
        (
            """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON a.id = c.id
            WHERE b.id IS NOT NULL
            """,
            [("bid", False), ("cid", True)],
        ),
        (
            """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON b.id = c.id OR a.id = 2
            WHERE c.id IS NOT NULL
            """,
            [("bid", True), ("cid", False)],
        ),
        (
            """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON COALESCE(b.id, 9) = c.id
            WHERE c.id IS NOT NULL
            """,
            [("bid", True), ("cid", False)],
        ),
        (
            """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id IS NOT DISTINCT FROM b.id
            LEFT JOIN c ON b.id = c.id
            WHERE c.id IS NOT NULL
            """,
            [("bid", True), ("cid", False)],
        ),
        (
            """
            SELECT a.id AS aid, b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON b.id = c.id
            WHERE c.v > 0
            """,
            [("aid", False), ("bid", False), ("cid", False)],
        ),
    ]

    # Join-side and ON-shape matrix for randomized execution. Every entry must
    # be executable against the three-table fixture.
    JOIN_SIDES = ["INNER", "LEFT", "RIGHT", "FULL", "CROSS"]
    ON_SHAPES = [
        "a.id = b.id",
        "a.id = b.id AND a.v = b.v",
        "a.id = b.id OR a.id = 2",
        "COALESCE(a.id, 9) = b.id",
        "a.id IS NOT DISTINCT FROM b.id",
    ]

    def setUp(self):
        import random

        self.conn = duckdb.connect()
        self.rng = random.Random(self.SEED)
        self.conn.execute("CREATE TABLE a (id INTEGER NOT NULL, v INTEGER)")
        self.conn.execute("CREATE TABLE b (id INTEGER NOT NULL, v INTEGER)")
        self.conn.execute("CREATE TABLE c (id INTEGER NOT NULL, v INTEGER)")
        self._populate("a", rows=12)
        self._populate("b", rows=12)
        self._populate("c", rows=12)

    def tearDown(self):
        self.conn.close()

    def _populate(self, table, rows):
        # Deliberately leave some tables empty on later repopulations and use a
        # small key domain so duplicate keys and unmatched rows are common.
        if rows == 0:
            return
        values = []
        for _ in range(rows):
            key = self.rng.randint(1, 6)
            v = None if self.rng.random() < 0.4 else self.rng.randint(-2, 5)
            values.append((key, v))
        self.conn.executemany(
            f"INSERT INTO {table} VALUES (?, ?)", values
        )

    def _repopulate(self, **sizes):
        for table in ("a", "b", "c"):
            self.conn.execute(f"DELETE FROM {table}")
        for table, size in sizes.items():
            self._populate(table, size)

    def _assert_no_false_null(self, sql, expected=None):
        inference = infer_nullability(
            sql,
            schema=self.SCHEMA,
            not_null=self.NOT_NULL,
            dialect=DIALECT,
        )
        if expected is not None:
            self.assertEqual(expected, inference, msg=sql)
        rows = self.conn.execute(sql).fetchall()
        for row in rows:
            self.assertEqual(len(row), len(inference), msg=sql)
            for position, ((name, nullable), value) in enumerate(
                zip(inference, row)
            ):
                if not nullable:
                    self.assertIsNotNone(
                        value,
                        msg=(
                            f"inferred non-NULL column {position} ('{name}') was "
                            f"NULL in: {sql}\ninference={inference}"
                        ),
                    )

    def test_handed_counterexamples_against_duckdb(self):
        # Exercise each reported shape against several data layouts, including
        # the exact small tables from the report.
        layouts = [
            # Exact report layout 1: a,b one matched row; c empty.
            {"a": 0, "b": 0, "c": 0},
        ]
        for sql, expected in self.HANDED_EXPECTATIONS:
            self._repopulate(a=0, b=0, c=0)
            self.conn.execute("INSERT INTO a VALUES (1, NULL)")
            self.conn.execute("INSERT INTO b VALUES (1, NULL)")
            with self.subTest(layout="report-1", sql=sql):
                self._assert_no_false_null(sql, expected)

            self._repopulate(a=0, b=0, c=0)
            self.conn.execute("INSERT INTO a VALUES (2, NULL)")
            self.conn.execute("INSERT INTO c VALUES (9, 1)")
            with self.subTest(layout="report-2", sql=sql):
                self._assert_no_false_null(sql, expected)

        # The report-1 layout genuinely yields the (1, NULL) result for q1.
        self._repopulate(a=0, b=0, c=0)
        self.conn.execute("INSERT INTO a VALUES (1, NULL)")
        self.conn.execute("INSERT INTO b VALUES (1, NULL)")
        q1 = self.HANDED_EXPECTATIONS[0][0]
        self.assertEqual([(1, None)], self.conn.execute(q1).fetchall())

        # And report-2 yields (NULL, 9) for q2.
        self._repopulate(a=0, b=0, c=0)
        self.conn.execute("INSERT INTO a VALUES (2, NULL)")
        self.conn.execute("INSERT INTO c VALUES (9, 1)")
        q2 = self.HANDED_EXPECTATIONS[1][0]
        self.assertEqual([(None, 9)], self.conn.execute(q2).fetchall())

    def test_nested_expression_where_against_duckdb(self):
        # c is reachable from the preserved side, so c exists while b stays
        # NULL-padded; the COALESCE predicate accepts that padded row.
        self._repopulate(a=0, b=0, c=0)
        self.conn.execute("INSERT INTO a VALUES (1, NULL)")
        self.conn.execute("INSERT INTO c VALUES (0, 9)")
        sql = """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON a.id = c.id OR 1 = 1
            WHERE COALESCE(b.id, 0) = c.id
        """
        inference = infer_nullability(
            sql,
            schema=self.SCHEMA,
            not_null=self.NOT_NULL,
            dialect=DIALECT,
        )
        self.assertEqual([("bid", True), ("cid", False)], inference)
        self.assertEqual([(None, 0)], self.conn.execute(sql).fetchall())

    def test_join_side_and_on_shape_matrix(self):
        where_variants = [
            "",
            "WHERE b.id IS NOT NULL",
            "WHERE c.id IS NOT NULL",
            "WHERE b.v > 0",
            "WHERE b.id IS NOT NULL OR c.id IS NOT NULL",
            "WHERE b.id IS NOT NULL AND c.id IS NOT NULL",
            # Nested-expression predicates: the outer expression is
            # NULL-rejecting, but a column buried in COALESCE/arithmetic does
            # not prove its source row is present.
            "WHERE COALESCE(b.id, 0) = c.id",
            "WHERE b.id + 1 = c.id",
            "WHERE NOT (COALESCE(b.id, 0) IS NULL)",
        ]
        for side in self.JOIN_SIDES:
            for on in self.ON_SHAPES:
                on_c = on.replace("a.id = b.id", "b.id = c.id").replace(
                    "a.v = b.v", "b.v = c.v"
                ).replace("a.id = 2", "b.id = 2").replace(
                    "COALESCE(a.id, 9)", "COALESCE(b.id, 9)"
                ).replace("a.id IS NOT DISTINCT FROM b.id", "b.id IS NOT DISTINCT FROM c.id")
                if side == "CROSS":
                    join_b = "CROSS JOIN b"
                    join_c = "CROSS JOIN c"
                else:
                    join_b = f"{side} JOIN b ON {on}"
                    join_c = f"{side} JOIN c ON {on_c}"
                for where in where_variants:
                    sql = (
                        "SELECT a.id AS aid, b.id AS bid, c.id AS cid "
                        f"FROM a {join_b} {join_c} {where}"
                    )
                    with self.subTest(side=side, on=on, where=where):
                        self._assert_no_false_null(sql)

    def test_right_full_and_chain_directions(self):
        queries = [
            "SELECT a.id AS aid, b.id AS bid FROM a RIGHT JOIN b ON a.id = b.id",
            "SELECT a.id AS aid, b.id AS bid FROM a FULL JOIN b ON a.id = b.id",
            "SELECT a.id AS aid, b.id AS bid FROM a FULL JOIN b ON a.id = b.id "
            "WHERE a.id IS NOT NULL",
            """
            SELECT a.id AS aid, b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON b.id = c.id
            """,
            """
            SELECT a.id AS aid, b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON b.id = c.id
            WHERE c.id IS NOT NULL
            """,
            # Reversed independent-join ordering, aligned by output names.
            """
            SELECT c.id AS cid, b.id AS bid
            FROM a
            LEFT JOIN c ON a.id = c.id
            LEFT JOIN b ON a.id = b.id
            WHERE b.id IS NOT NULL
            """,
        ]
        for sql in queries:
            with self.subTest(sql=sql):
                self._assert_no_false_null(sql)

    def test_cte_derived_and_alias_shadowing_execution(self):
        queries = [
            """
            WITH d AS (
                SELECT b.id AS bid, c.id AS cid
                FROM a
                LEFT JOIN b ON a.id = b.id
                LEFT JOIN c ON b.id = c.id OR a.id = 2
                WHERE c.id IS NOT NULL
            )
            SELECT bid, cid FROM d
            """,
            """
            SELECT bid, cid FROM (
                SELECT b.id AS bid, c.id AS cid
                FROM a
                LEFT JOIN b ON a.id = b.id
                LEFT JOIN c ON a.id = c.id
                WHERE b.id IS NOT NULL
            ) d
            """,
            """
            SELECT b.id AS outer_bid, d.x AS inner_bid
            FROM a
            LEFT JOIN b ON a.id = b.id
            JOIN (
                SELECT b2.id AS x
                FROM a AS a2
                LEFT JOIN b AS b2 ON a2.id = b2.id
                WHERE b2.id IS NOT NULL
            ) d ON d.x = a.id
            """,
        ]
        for sql in queries:
            with self.subTest(sql=sql):
                self._assert_no_false_null(sql)

    def test_empty_tables_and_repeated_runs_are_stable(self):
        # Fully empty fixture: outer joins still emit NULL-padded rows.
        self._repopulate(a=0, b=0, c=0)
        sql = """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON a.id = c.id
            WHERE b.id IS NOT NULL
        """
        first = infer_nullability(
            sql, schema=self.SCHEMA, not_null=self.NOT_NULL, dialect=DIALECT
        )
        second = infer_nullability(
            sql, schema=self.SCHEMA, not_null=self.NOT_NULL, dialect=DIALECT
        )
        self.assertEqual(first, second)
        self.assertEqual([], self.conn.execute(sql).fetchall())

        # Repopulate and confirm a non-NULL claim still never observes a NULL.
        self._populate("a", 8)
        self._populate("c", 8)  # b stays empty
        self._assert_no_false_null(
            """
            SELECT b.id AS bid, c.id AS cid
            FROM a
            LEFT JOIN b ON a.id = b.id
            LEFT JOIN c ON a.id = c.id
            WHERE c.id IS NOT NULL
            """
        )
