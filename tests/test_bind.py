import sqlite3
import unittest
from decimal import Decimal

import duckdb

import sqlglot

from sqlglot import BoundSQL, exp, parse_one, transpile, transpile_bound
from sqlglot.errors import BindError, ErrorLevel, ParseError, UnsupportedError

SOURCE_STYLES = ("qmark", "named", "numeric")
TARGET_STYLES = ("qmark", "named", "numeric")


class Opaque:
    """A value that refuses to be stringified, compared, hashed or copied."""

    def __str__(self):
        raise AssertionError("parameter value was stringified")

    def __repr__(self):
        raise AssertionError("parameter value was repr'd")

    def __eq__(self, other):
        raise AssertionError("parameter value was compared")

    def __ne__(self, other):
        raise AssertionError("parameter value was compared")

    def __hash__(self):
        raise AssertionError("parameter value was hashed")

    def __copy__(self):
        raise AssertionError("parameter value was copied")

    def __deepcopy__(self, memo):
        raise AssertionError("parameter value was deep-copied")


def source_query(source_style, values):
    """Returns a query and containers for a source style, using the given values."""
    if source_style == "qmark":
        sql = "SELECT " + ", ".join("?" for _ in values)
        return sql, list(values)
    if source_style == "named":
        names = [f"v{i}" for i in range(len(values))]
        sql = "SELECT " + ", ".join(f":{name}" for name in names)
        return sql, dict(zip(names, values))
    sql = "SELECT " + ", ".join(f"${i + 1}" for i in range(len(values)))
    return sql, list(values)


class TestTranspileBound(unittest.TestCase):
    def test_bound_attributes(self):
        result = transpile_bound(
            "SELECT ?, ?", ["a", "b"], read="duckdb", write="mysql", source_style="qmark"
        )
        self.assertIsInstance(result, BoundSQL)
        self.assertEqual(result.sql, "SELECT ?, ?")
        self.assertEqual(result.parameters, ("a", "b"))
        self.assertEqual(result.parameter_order, (1, 2))
        self.assertEqual(tuple(result), ("SELECT ?, ?", ("a", "b"), (1, 2)))
        # Diagnostics carry SQL and identities, but never parameter values.
        self.assertNotIn("'a'", repr(result))
        self.assertNotIn("'b'", repr(result))

    def test_all_style_combinations(self):
        values = ["zero", "one", "two"]
        expected_order = {
            "qmark": (1, 2, 3),
            "numeric": (1, 2, 3),
            "named": ("v0", "v1", "v2"),
        }

        for source_style in SOURCE_STYLES:
            sql, parameters = source_query(source_style, values)
            for target_style in TARGET_STYLES:
                with self.subTest(source=source_style, target=target_style):
                    expected_sql = "SELECT " + ", ".join(
                        "?"
                        if target_style == "qmark"
                        else (f"${i + 1}" if target_style == "numeric" else f":p{i + 1}")
                        for i in range(3)
                    )
                    expected_parameters = (
                        tuple(values)
                        if target_style != "named"
                        else {f"p{i + 1}": value for i, value in enumerate(values)}
                    )

                    result = transpile_bound(
                        sql,
                        parameters,
                        read="postgres",
                        write="postgres",
                        source_style=source_style,
                        target_style=target_style,
                    )
                    self.assertEqual(result.sql, expected_sql)
                    self.assertEqual(result.parameters, expected_parameters)
                    self.assertIs(
                        type(result.parameters), dict if target_style == "named" else tuple
                    )
                    self.assertEqual(result.parameter_order, expected_order[source_style])
                    self.assertIs(type(result.parameter_order), tuple)

    def test_unused_numeric_inputs_can_be_passed_as_gap(self):
        # $3 and $1 are used; $2 is a gap that must not reach the target binding.
        result = transpile_bound(
            "SELECT $3, $1",
            ["first", "unused-gap", "third"],
            read="postgres",
            write="postgres",
            source_style="numeric",
            target_style="numeric",
        )
        self.assertEqual(result.sql, "SELECT $1, $2")
        self.assertEqual(result.parameters, ("third", "first"))
        self.assertEqual(result.parameter_order, (3, 1))

    def test_numeric_repeated_index(self):
        result = transpile_bound(
            "SELECT $2, $1, $2",
            ["first", "second"],
            read="postgres",
            write="duckdb",
            source_style="numeric",
            target_style="numeric",
        )
        self.assertEqual(result.sql, "SELECT $1, $2, $1")
        self.assertEqual(result.parameters, ("second", "first"))
        self.assertEqual(result.parameter_order, (2, 1))

    def test_named_repeated_name(self):
        result = transpile_bound(
            "SELECT :a, :b, :a",
            {"a": "A", "b": "B"},
            read="postgres",
            write="postgres",
            source_style="named",
            target_style="numeric",
        )
        self.assertEqual(result.sql, "SELECT $1, $2, $1")
        self.assertEqual(result.parameters, ("A", "B"))
        self.assertEqual(result.parameter_order, ("a", "b"))

    def test_values_with_equal_contents_have_distinct_identities(self):
        first = [1, 2]
        second = [1, 2]
        result = transpile_bound(
            "SELECT ?, ?", [first, second], read="duckdb", write="duckdb", source_style="qmark"
        )
        self.assertEqual(result.parameter_order, (1, 2))
        self.assertIs(result.parameters[0], first)
        self.assertIs(result.parameters[1], second)

    def test_expression_reordering(self):
        # MySQL's LOCATE(haystack, needle) swaps DuckDB's STRPOS(haystack, needle).
        result = transpile_bound(
            "SELECT STRPOS(?, ?)",
            ["haystack", "needle"],
            read="duckdb",
            write="mysql",
            source_style="qmark",
        )
        self.assertEqual(result.sql, "SELECT LOCATE(?, ?)")
        self.assertEqual(result.parameters, ("needle", "haystack"))
        self.assertEqual(result.parameter_order, (2, 1))

        # PostgreSQL's POSITION(needle IN haystack) swaps as well.
        result = transpile_bound(
            "SELECT STRPOS(?, ?)",
            ["haystack", "needle"],
            read="duckdb",
            write="postgres",
            source_style="qmark",
        )
        self.assertEqual(result.sql, "SELECT POSITION(? IN ?)")
        self.assertEqual(result.parameters, ("needle", "haystack"))

    def test_expression_copying(self):
        # MySQL emulates NULLS LAST by repeating the projected parameter in ORDER BY.
        result = transpile_bound(
            "SELECT ? AS a, ? AS b ORDER BY a NULLS LAST",
            ["a", "b"],
            read="duckdb",
            write="mysql",
            source_style="qmark",
        )
        self.assertEqual(
            result.sql,
            "SELECT ? AS a, ? AS b ORDER BY CASE WHEN ? IS NULL THEN 1 ELSE 0 END, ?",
        )
        self.assertEqual(result.parameters, ("a", "b", "a", "a"))
        self.assertEqual(result.parameter_order, (1, 2, 1, 1))

    def test_expression_disappearing(self):
        # Hive drops APPROX_DISTINCT's accuracy argument, so that parameter has no target slot.
        result = transpile_bound(
            "SELECT APPROX_DISTINCT(?, ?)",
            ["expr", "accuracy"],
            read="presto",
            write="hive",
            source_style="qmark",
            unsupported_level=ErrorLevel.IGNORE,
        )
        self.assertEqual(result.sql, "SELECT APPROX_COUNT_DISTINCT(?)")
        self.assertEqual(result.parameters, ("expr",))
        self.assertEqual(result.parameter_order, (1,))

        # The input is still validated against the original statement.
        with self.assertRaises(BindError):
            transpile_bound(
                "SELECT APPROX_DISTINCT(?, ?)",
                ["expr"],
                read="presto",
                write="hive",
                source_style="qmark",
                unsupported_level=ErrorLevel.IGNORE,
            )

    def test_composite_queries(self):
        queries = [
            "WITH c(x) AS (SELECT ?) SELECT ?, (SELECT ? FROM c) FROM c WHERE x = ?",
            "SELECT * FROM (SELECT ? AS a) t UNION ALL SELECT ? FROM t",
            "INSERT INTO t SELECT ?, ? FROM u",
            "SELECT SUM(?) OVER (PARTITION BY ?) FROM t",
        ]
        for sql in queries:
            count = sql.count("?")
            with self.subTest(sql=sql):
                result = transpile_bound(
                    sql,
                    list(range(1, count + 1)),
                    read="duckdb",
                    write="mysql",
                    source_style="qmark",
                )
                self.assertEqual(result.parameter_order, tuple(range(1, count + 1)))
                self.assertEqual(result.sql.count("?"), count)
                self.assertEqual(result.parameters, tuple(range(1, count + 1)))

    def test_formatting_options_do_not_affect_binding(self):
        sql = "SELECT ? AS a, ? AS b FROM t ORDER BY a NULLS LAST"
        baseline = transpile_bound(
            sql, ["a", "b"], read="duckdb", write="mysql", source_style="qmark"
        )
        for opts in (
            {"pretty": True},
            {"identify": True},
            {"comments": False},
            {"pretty": True, "identify": True},
        ):
            with self.subTest(**opts):
                result = transpile_bound(
                    sql, ["a", "b"], read="duckdb", write="mysql", source_style="qmark", **opts
                )
                self.assertEqual(result.parameter_order, baseline.parameter_order)
                self.assertEqual(result.parameters, baseline.parameters)
                self.assertEqual(result.sql.count("?"), 4)

    def test_lexical_interference(self):
        result = transpile_bound(
            "SELECT ?, '?', \"?\", `?`, /* ? */ ? -- ?\n",
            ["one", "two"],
            read="mysql",
            write="mysql",
            source_style="qmark",
        )
        self.assertEqual(result.parameter_order, (1, 2))
        self.assertEqual(result.parameters, ("one", "two"))
        # Only the real markers survive as placeholders; literal text stays literal.
        parsed = parse_one(result.sql, read="mysql")
        self.assertEqual(sum(1 for node in parsed.walk() if isinstance(node, exp.Placeholder)), 2)

        # PostgreSQL dollar-quoted strings and :: casts coexist with parameters.
        result = transpile_bound(
            "SELECT $$ ? $$, $tag$ ? :a $tag$, ?::INT",
            ["value"],
            read="postgres",
            write="postgres",
            source_style="qmark",
        )
        self.assertEqual(result.parameter_order, (1,))
        self.assertEqual(result.parameters, ("value",))

        result = transpile_bound(
            "SELECT $$ ? $$, :a",
            {"a": "named"},
            read="postgres",
            write="postgres",
            source_style="named",
        )
        self.assertEqual(result.parameter_order, ("a",))
        self.assertEqual(result.parameters, ("named",))

    def test_postgres_json_question_mark_operator(self):
        # `data ? 'key'` is the JSON containment operator, while the trailing ? is a parameter.
        result = transpile_bound(
            "SELECT data ? 'key', ?",
            ["value"],
            read="postgres",
            write="postgres",
            source_style="qmark",
        )
        self.assertEqual(result.sql, "SELECT data ? 'key', ?")
        self.assertEqual(result.parameters, ("value",))
        self.assertEqual(result.parameter_order, (1,))

    def test_dialect_path_operators_are_not_named_parameters(self):
        # Snowflake `col:key` is a path accessor, not a parameter.
        result = transpile_bound(
            "SELECT col:key, :a FROM t",
            {"a": "value"},
            read="snowflake",
            write="snowflake",
            source_style="named",
        )
        self.assertEqual(result.parameter_order, ("a",))
        self.assertEqual(result.parameters, ("value",))
        self.assertNotIn(":a", result.sql)

        # DuckDB `arr[:i]` is a slice, not a parameter.
        result = transpile_bound(
            "SELECT arr[:i], :a",
            {"a": 1},
            read="duckdb",
            write="duckdb",
            source_style="named",
        )
        self.assertEqual(result.parameter_order, ("a",))
        self.assertEqual(result.parameters, (1,))

    def test_named_markers_dialects_that_need_rewriting(self):
        # DuckDB parses `:a` as a prefix alias at the start of a projection.
        result = transpile_bound(
            "SELECT :a AS x, :b",
            {"a": 1, "b": 2},
            read="duckdb",
            write="postgres",
            source_style="named",
        )
        self.assertEqual(result.sql, "SELECT ? AS x, ?")
        self.assertEqual(result.parameter_order, ("a", "b"))
        self.assertEqual(result.parameters, (1, 2))

    def test_dialect_instances_and_delegating_generators(self):
        read = sqlglot.Dialect.get_or_raise("duckdb")
        write = sqlglot.Dialect.get_or_raise("mysql")
        result = transpile_bound(
            "SELECT STRPOS(?, ?)", ["hay", "needle"], read=read, write=write, source_style="qmark"
        )
        self.assertEqual(result.sql, "SELECT LOCATE(?, ?)")
        self.assertEqual(result.parameter_order, (2, 1))

        # Athena's generator delegates to internal Hive/Trino generators, which must
        # still emit resolvable markers.
        result = transpile_bound(
            "SELECT ?, ?", ["a", "b"], read="duckdb", write="athena", source_style="qmark"
        )
        self.assertEqual(result.sql, "SELECT ?, ?")
        self.assertEqual(result.parameters, ("a", "b"))
        self.assertEqual(result.parameter_order, (1, 2))

    def test_value_passthrough(self):
        first, second = Opaque(), Opaque()
        values = [b"bytes", None, Decimal("1.10"), first, second]
        result = transpile_bound(
            "SELECT $1, $2, $3, $4, $5",
            values,
            read="postgres",
            write="duckdb",
            source_style="numeric",
            target_style="numeric",
        )
        self.assertEqual(result.parameter_order, (1, 2, 3, 4, 5))
        for i, value in enumerate(values):
            self.assertIs(result.parameters[i], value)
        self.assertIsInstance(result.parameters[2], Decimal)

        # Repeated use preserves object identity.
        result = transpile_bound(
            "SELECT $1, $2, $1",
            [first, second],
            read="postgres",
            write="mysql",
            source_style="numeric",
        )
        self.assertEqual(result.parameter_order, (1, 2, 1))
        self.assertIs(result.parameters[0], first)
        self.assertIs(result.parameters[1], second)
        self.assertIs(result.parameters[2], first)

    def test_inputs_are_not_mutated(self):
        values = [Opaque(), Opaque()]
        original_values = list(values)
        result = transpile_bound(
            "SELECT ?, ?", values, read="duckdb", write="mysql", source_style="qmark"
        )
        self.assertEqual(len(values), 2)
        self.assertIs(values[0], original_values[0])
        self.assertIs(values[1], original_values[1])
        self.assertIs(result.parameters[0], values[0])

        mapping = {"a": Opaque(), "b": Opaque()}
        original_mapping = dict(mapping)
        result = transpile_bound(
            "SELECT :a, :b, :a",
            mapping,
            read="postgres",
            write="mysql",
            source_style="named",
            target_style="named",
        )
        self.assertEqual(set(mapping), {"a", "b"})
        self.assertIs(mapping["a"], original_mapping["a"])
        self.assertIs(mapping["b"], original_mapping["b"])
        self.assertIs(result.parameters["p1"], mapping["a"])

    def test_call_isolation(self):
        first = transpile_bound(
            "SELECT ?, ?", ["a", "b"], read="duckdb", write="mysql", source_style="qmark"
        )
        # A failing call in between must not leak any state.
        with self.assertRaises(BindError):
            transpile_bound("SELECT ?", [], read="duckdb", write="mysql", source_style="qmark")
        with self.assertRaises(ParseError):
            transpile_bound(
                "SELECT FROM", ["x"], read="duckdb", write="mysql", source_style="qmark"
            )
        second = transpile_bound(
            "SELECT ?, ?", ["c", "d"], read="duckdb", write="mysql", source_style="qmark"
        )
        self.assertEqual(first.parameters, ("a", "b"))
        self.assertEqual(second.parameters, ("c", "d"))
        self.assertEqual(second.parameter_order, (1, 2))

    def test_no_parameters(self):
        for source_style, parameters in (
            ("qmark", []),
            ("named", {}),
            ("numeric", ()),
        ):
            for target_style in TARGET_STYLES:
                with self.subTest(source=source_style, target=target_style):
                    result = transpile_bound(
                        "SELECT 1",
                        parameters,
                        read="duckdb",
                        write="duckdb",
                        source_style=source_style,
                        target_style=target_style,
                    )
                    self.assertEqual(result.sql, "SELECT 1")
                    self.assertEqual(result.parameters, {} if target_style == "named" else ())
                    self.assertEqual(result.parameter_order, ())

    def test_single_trailing_semicolon(self):
        result = transpile_bound(
            "SELECT ?;", ["a"], read="duckdb", write="duckdb", source_style="qmark"
        )
        self.assertEqual(result.sql, "SELECT ?")

    def test_bind_errors(self):
        error_cases = [
            # empty / multiple statements
            ("", [], "duckdb", "qmark", "qmark"),
            ("   ", [], "duckdb", "qmark", "qmark"),
            (";", [], "duckdb", "qmark", "qmark"),
            ("SELECT 1; SELECT 2", [], "duckdb", "qmark", "qmark"),
            ("SELECT 1;;", [], "duckdb", "qmark", "qmark"),
            # unknown styles
            ("SELECT ?", [], "duckdb", "question", "qmark"),
            ("SELECT ?", [], "duckdb", "qmark", "question"),
            # missing / extra parameters
            ("SELECT ?, ?", [], "duckdb", "qmark", "qmark"),
            ("SELECT ?, ?", ["a", "b", "c"], "duckdb", "qmark", "qmark"),
            ("SELECT ?", {"a": 1}, "duckdb", "qmark", "qmark"),
            ("SELECT :a", {}, "postgres", "named", "qmark"),
            ("SELECT :a", {"a": 1, "b": 2}, "postgres", "named", "qmark"),
            ("SELECT :a", {"b": 1}, "postgres", "named", "qmark"),
            ("SELECT $2", ["a"], "postgres", "numeric", "qmark"),
            # invalid numeric indices
            ("SELECT $0", ["a"], "postgres", "numeric", "qmark"),
            ("SELECT $01", ["a"], "postgres", "numeric", "qmark"),
            ("SELECT $ 1", ["a"], "postgres", "numeric", "qmark"),
            # invalid named markers
            ("SELECT :1", {}, "snowflake", "named", "qmark"),
            # mixed source styles
            ("SELECT ?, :a", {"a": 1}, "postgres", "named", "qmark"),
            ("SELECT ?, $1", ["a", "b"], "postgres", "qmark", "qmark"),
            ("SELECT :a, $1", {"a": 1}, "postgres", "named", "qmark"),
            ("SELECT ?", ["a"], "postgres", "numeric", "qmark"),
        ]

        for sql, parameters, read, source_style, target_style in error_cases:
            with (
                self.subTest(sql=sql, source=source_style, target=target_style),
                self.assertRaises(BindError),
            ):
                transpile_bound(
                    sql,
                    parameters,
                    read=read,
                    write=read,
                    source_style=source_style,
                    target_style=target_style,
                )

    def test_error_diagnostics_do_not_leak_values(self):
        # Opaque resists repr/str/eq/hash, so these calls only succeed if the
        # diagnostics never touch a parameter value.
        secret = Opaque()
        cases = [
            ("SELECT ?, ?", [secret], "duckdb", "qmark"),
            ("SELECT :a", {"b": secret}, "postgres", "named"),
            ("SELECT $2", [secret], "postgres", "numeric"),
            ("SELECT $2", {"b": secret}, "postgres", "numeric"),
        ]

        for sql, parameters, read, source_style in cases:
            with self.subTest(sql=sql, source=source_style):
                with self.assertRaises(BindError) as context:
                    transpile_bound(
                        sql,
                        parameters,
                        read=read,
                        write=read,
                        source_style=source_style,
                    )
                self.assertIsInstance(context.exception, BindError)

        # A value whose repr is benign still never appears in the message.
        class Marker:
            def __repr__(self):
                return "LEAKED_PARAMETER_VALUE"

            def __str__(self):
                return "LEAKED_PARAMETER_VALUE"

        for sql, parameters, read, source_style in (
            ("SELECT ?, ?", [Marker()], "duckdb", "qmark"),
            ("SELECT :a", {"b": Marker()}, "postgres", "named"),
            ("SELECT $2", [Marker()], "postgres", "numeric"),
        ):
            with self.subTest(sql=sql, source=source_style):
                with self.assertRaises(BindError) as context:
                    transpile_bound(
                        sql, parameters, read=read, write=read, source_style=source_style
                    )
                self.assertNotIn("LEAKED_PARAMETER_VALUE", str(context.exception))

    def test_named_target_with_reordered_duplicates(self):
        result = transpile_bound(
            "WITH c(x) AS (SELECT :a) SELECT :b, (SELECT :a FROM c), :b FROM c",
            {"a": "A", "b": "B"},
            read="postgres",
            write="postgres",
            source_style="named",
            target_style="named",
        )
        self.assertEqual(
            result.sql,
            "WITH c(x) AS (SELECT :p1) SELECT :p2, (SELECT :p1 FROM c), :p2 FROM c",
        )
        self.assertEqual(result.parameters, {"p1": "A", "p2": "B"})
        # Slot-based styles list each slot once, even if the marker repeats.
        self.assertEqual(result.parameter_order, ("a", "b"))

    def test_syntax_errors_remain_parse_errors(self):
        with self.assertRaises(ParseError):
            transpile_bound("SELECT FROM", [], read="duckdb", write="duckdb", source_style="qmark")

    def test_unsupported_errors_follow_existing_levels(self):
        sql = "SELECT APPROX_DISTINCT(?, ?)"
        with self.assertRaises(UnsupportedError):
            transpile_bound(
                sql,
                ["expr", "accuracy"],
                read="presto",
                write="hive",
                source_style="qmark",
                unsupported_level=ErrorLevel.RAISE,
            )
        # The default level only warns and still produces a result.
        result = transpile_bound(
            sql,
            ["expr", "accuracy"],
            read="presto",
            write="hive",
            source_style="qmark",
        )
        self.assertEqual(result.parameter_order, (1,))
        self.assertEqual(result.parameters, ("expr",))

    def test_error_level_is_forwarded_to_parser(self):
        with self.assertRaises(ParseError):
            transpile_bound(
                "SELECT FROM",
                [],
                read="duckdb",
                write="duckdb",
                source_style="qmark",
                error_level=ErrorLevel.RAISE,
            )

    def test_existing_transpile_unchanged(self):
        self.assertEqual(transpile("SELECT ?", read="duckdb", write="mysql"), ["SELECT ?"])
        self.assertEqual(
            transpile("SELECT $1, $2", read="postgres", write="mysql"), ["SELECT @1, @2"]
        )


class TestTranspileBoundExecution(unittest.TestCase):
    """Executes the generated SQL and parameters against real drivers."""

    def test_sqlite_round_trips(self):
        cases = [
            (
                "SELECT :a + :b AS total, :a * :b AS product",
                {"a": 7, "b": 6},
                "named",
                ("qmark", "numeric", "named"),
            ),
            (
                "SELECT $2 - $1 AS diff, $1 AS first",
                [4, 9],
                "numeric",
                ("qmark", "numeric", "named"),
            ),
            (
                "SELECT ? || ? AS joined",
                ["foo", "bar"],
                "qmark",
                ("qmark", "numeric", "named"),
            ),
        ]

        for sql, parameters, source_style, target_styles in cases:
            with sqlite3.connect(":memory:") as connection:
                expected = connection.execute(
                    sql, _sqlite_source_parameters(source_style, parameters)
                ).fetchall()

            for target_style in target_styles:
                with self.subTest(sql=sql, source=source_style, target=target_style):
                    result = transpile_bound(
                        sql,
                        parameters,
                        read="sqlite",
                        write="sqlite",
                        source_style=source_style,
                        target_style=target_style,
                    )
                    with sqlite3.connect(":memory:") as connection:
                        actual = connection.execute(result.sql, result.parameters).fetchall()
                    self.assertEqual(actual, expected)

    def test_duckdb_round_trips(self):
        cases = [
            ("SELECT ? - ? AS diff", [10, 3], "qmark", ("qmark", "numeric")),
            ("SELECT $2 - $1 AS diff", [3, 10], "numeric", ("qmark", "numeric")),
            ("SELECT STRPOS(?, ?) AS pos", ["hello", "ll"], "qmark", ("qmark", "numeric")),
        ]

        for sql, parameters, source_style, target_styles in cases:
            with duckdb.connect() as connection:
                expected = connection.execute(sql, parameters).fetchall()

            for target_style in target_styles:
                with self.subTest(sql=sql, source=source_style, target=target_style):
                    result = transpile_bound(
                        sql,
                        parameters,
                        read="duckdb",
                        write="duckdb",
                        source_style=source_style,
                        target_style=target_style,
                    )
                    with duckdb.connect() as connection:
                        actual = connection.execute(result.sql, result.parameters).fetchall()
                    self.assertEqual(actual, expected)

    def test_duckdb_to_sqlite_strpos_swap(self):
        sql = "SELECT STRPOS(?, ?) AS pos"
        parameters = ["haystack", "stack"]
        with duckdb.connect() as connection:
            expected = connection.execute(sql, parameters).fetchall()

        result = transpile_bound(
            sql, parameters, read="duckdb", write="sqlite", source_style="qmark"
        )
        self.assertEqual(result.sql, "SELECT INSTR(?, ?) AS pos")
        self.assertEqual(result.parameters, ("haystack", "stack"))
        self.assertEqual(result.parameter_order, (1, 2))
        with sqlite3.connect(":memory:") as connection:
            actual = connection.execute(result.sql, result.parameters).fetchall()
        self.assertEqual(actual, expected)

    def test_mysql_parameter_reordering_and_nulls_last(self):
        # Checked against independent expectations rather than executing MySQL.
        result = transpile_bound(
            "SELECT STRPOS(?, ?)",
            ["hay", "needle"],
            read="duckdb",
            write="mysql",
            source_style="qmark",
        )
        self.assertEqual(result.sql, "SELECT LOCATE(?, ?)")
        self.assertEqual(result.parameters, ("needle", "hay"))
        self.assertEqual(result.parameter_order, (2, 1))

        result = transpile_bound(
            "SELECT ? AS a, ? AS b ORDER BY a NULLS LAST",
            ["first", "second"],
            read="duckdb",
            write="mysql",
            source_style="qmark",
        )
        self.assertEqual(
            result.sql,
            "SELECT ? AS a, ? AS b ORDER BY CASE WHEN ? IS NULL THEN 1 ELSE 0 END, ?",
        )
        self.assertEqual(result.parameters, ("first", "second", "first", "first"))
        self.assertEqual(result.parameter_order, (1, 2, 1, 1))

        # Named source with the same shape keeps identities rather than values.
        result = transpile_bound(
            "SELECT :a AS a, :b AS b ORDER BY a NULLS LAST",
            {"a": "first", "b": "second"},
            read="postgres",
            write="mysql",
            source_style="named",
            target_style="named",
        )
        self.assertEqual(
            result.sql,
            "SELECT :p1 AS a, :p2 AS b ORDER BY CASE WHEN :p1 IS NULL THEN 1 ELSE 0 END, :p1",
        )
        self.assertEqual(result.parameters, {"p1": "first", "p2": "second"})
        self.assertEqual(result.parameter_order, ("a", "b"))


def _sqlite_source_parameters(source_style, parameters):
    if source_style == "numeric":
        # sqlite binds a positional sequence by first occurrence for `$n`, but binds
        # an index-keyed mapping by number, which is the source semantics we test.
        return {str(index): value for index, value in enumerate(parameters, start=1)}
    return parameters


if __name__ == "__main__":
    unittest.main()
