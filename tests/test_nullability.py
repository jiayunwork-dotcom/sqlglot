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
