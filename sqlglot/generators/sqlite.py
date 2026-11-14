from __future__ import annotations

import datetime
import re
import typing as t

from sqlglot import exp, generator, transforms
from sqlglot.dialects.dialect import (
    any_value_to_max_sql,
    arrow_json_extract_sql,
    count_if_to_sum,
    groupconcat_sql,
    no_ilike_sql,
    no_pivot_sql,
    no_tablesample_sql,
    rename_func,
    strposition_sql,
)
from sqlglot.generator import unsupported_args
from sqlglot.tokens import TokenType

# DuckDB-lenient datetime text: '2024-1-5', '2024/01/15', '2024-01-15T10:30:00',
# '2024-01-15 1:2:3', '2024-01-15 10:30:45.123', surrounding whitespace.
_TEMPORAL_LITERAL_RE = re.compile(
    r"^(\d{1,4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{1,2})(?::(\d{1,2})(\.\d{1,9})?)?)?$"
)

# Text that SQLite's own date/time functions accept as-is.
_SQLITE_DATE_TEXT_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?$"
)


def _normalize_temporal_literal(
    text: str, want: str
) -> "str | tuple[str, str] | None":
    """Parse a string literal the way DuckDB's ``CAST`` to ``DATE``/``TIMESTAMP`` would.

    Args:
        text: the literal's contents.
        want: ``"date"`` or ``"timestamp"``.

    Returns:
        For ``want == "date"``: the canonical ``YYYY-MM-DD`` string, or ``None`` if the
        literal is not a valid DuckDB date. For ``want == "timestamp"``: a
        ``(canonical_without_fraction, fraction_suffix)`` tuple, or ``None``.
    """
    normalized = text.strip().replace("/", "-")
    match = _TEMPORAL_LITERAL_RE.match(normalized)
    if not match:
        return None

    year, month, day = (int(match.group(i)) for i in (1, 2, 3))
    hour = int(match.group(4) or 0)
    minute = int(match.group(5) or 0)
    second = int(match.group(6) or 0)
    fraction = match.group(7) or ""

    try:
        value = datetime.date(year, month, day)
    except ValueError:
        return None

    if (hour, minute, second) == (24, 0, 0):
        # DuckDB rolls '24:00:00' over to the next day
        value += datetime.timedelta(days=1)
        hour = 0
    elif hour > 23 or minute > 59 or second > 59:
        return None

    date_part = f"{value.year:04d}-{value.month:02d}-{value.day:02d}"
    if want == "date":
        return date_part
    return f"{date_part} {hour:02d}:{minute:02d}:{second:02d}", fraction


def _transform_create(expression: exp.Expr) -> exp.Expr:
    """Move primary key to a column and place auto_increment after it."""
    schema = expression.this

    if isinstance(expression, exp.Create) and isinstance(schema, exp.Schema):
        defs = {}
        primary_key = None

        for e in schema.expressions:
            if isinstance(e, exp.ColumnDef):
                defs[e.name] = e
            elif isinstance(e, exp.PrimaryKey):
                primary_key = e

        if primary_key and len(primary_key.expressions) == 1:
            column = defs[primary_key.expressions[0].name]
            column.append(
                "constraints", exp.ColumnConstraint(kind=exp.PrimaryKeyColumnConstraint())
            )
            schema.expressions.remove(primary_key)

        for column in defs.values():
            primary_key_index = -1
            auto_increment = None
            auto_increment_index = -1

            for i, constraint in enumerate(column.constraints):
                if isinstance(constraint.kind, exp.PrimaryKeyColumnConstraint):
                    primary_key_index = i
                elif isinstance(constraint.kind, exp.AutoIncrementColumnConstraint):
                    auto_increment = constraint
                    auto_increment_index = i

            if auto_increment is not None and 0 <= auto_increment_index < primary_key_index:
                column.constraints.remove(auto_increment)
                column.constraints.insert(primary_key_index, auto_increment)

    return expression


def _generated_to_auto_increment(expression: exp.Expr) -> exp.Expr:
    if not isinstance(expression, exp.ColumnDef):
        return expression

    generated = expression.find(exp.GeneratedAsIdentityColumnConstraint)

    # Only rewrite true identity columns. Expression-bearing forms are computed
    # columns (GENERATED ALWAYS AS (expr)) and must keep their expression.
    if generated and generated.expression is None:
        t.cast(exp.ColumnConstraint, generated.parent).pop()

        not_null = expression.find(exp.NotNullColumnConstraint)
        if not_null:
            t.cast(exp.ColumnConstraint, not_null.parent).pop()

        expression.append(
            "constraints", exp.ColumnConstraint(kind=exp.AutoIncrementColumnConstraint())
        )

    return expression


class SQLiteGenerator(generator.Generator):
    """Generates SQL for SQLite.

    DuckDB-compatible date/time semantics
    =====================================

    SQLite has no native ``DATE``/``TIMESTAMP`` types, so temporal values are
    represented as ISO-8601 text (the same representation SQLite's own date and
    time functions produce and accept):

    - ``DATE``      -> ``'YYYY-MM-DD'``
    - ``TIMESTAMP`` -> ``'YYYY-MM-DD HH:MM:SS'`` (fractional seconds preserved
      when present)
    - ``BOOLEAN``   -> ``0`` / ``1``

    The following DuckDB date/time constructs are transpiled to semantically
    equivalent SQLite (only built-in functions available in SQLite >= 3.39 are
    used, so the output runs in Python's stdlib ``sqlite3``):

    - ``x +/- INTERVAL n unit`` and DuckDB's ``DATE_ADD(x, INTERVAL ...)`` for
      units YEAR/QUARTER/MONTH/WEEK/DAY/HOUR/MINUTE/SECOND. The result is
      always ``TIMESTAMP`` text, matching DuckDB. Month-family units clamp to
      the last day of the target month (``2024-01-31 + 1 month = 2024-02-29``),
      which SQLite's own ``months`` modifier gets wrong (it overflows into the
      next month), so they are computed via first-of-month arithmetic instead.
      ``DATE +/- <integer>`` adds/subtracts days and stays a ``DATE``.
    - ``DATE_TRUNC(unit, x)`` -> ``TIMESTAMP`` text (DuckDB returns ``TIMESTAMP``
      even for ``DATE`` input).
    - ``DATE_DIFF``/``DATEDIFF(unit, a, b)``: unit *boundaries* crossed between
      the two values, except ``WEEK`` which counts whole 7-day spans, matching
      DuckDB.
    - ``EXTRACT(part FROM x)`` / ``DATE_PART(part, x)`` and the synonym
      functions ``YEAR``, ``MONTH``, ``DAY``, ``DAYOFWEEK``, ``DAYOFYEAR``,
      ``WEEK``, ``QUARTER`` (also ``HOUR``/``MINUTE``/``SECOND``). ``WEEK`` is
      the ISO-8601 week and ``ISOYEAR`` its year; ``EPOCH`` yields seconds
      since 1970-01-01 as a REAL, like DuckDB's ``DOUBLE``.
    - ``LAST_DAY(x)`` -> ``DATE`` text.
    - ``CAST``/``TRY_CAST`` between ``VARCHAR``, ``DATE`` and ``TIMESTAMP``.
      DuckDB-lenient input (unpadded components, ``/`` separators, ``T``
      separator, ``24:00:00``) is normalized; invalid text yields ``NULL``
      (DuckDB's ``TRY_CAST`` semantics; DuckDB's ``CAST`` would raise, which
      SQLite cannot do). ``DATE``/``TIMESTAMP`` comparisons (=, <>, <, <=, >,
      >=, BETWEEN) are normalized to a common representation.

    Type information is taken from annotations (e.g. via
    ``sqlglot.optimizer.annotate_types``) when available and otherwise inferred
    from literals and ``CAST``s. When a temporal operand's type cannot be
    determined and the correct translation depends on it, the construct is
    reported as unsupported instead of silently generating wrong SQL.

    Not supported (reported via ``unsupported``): ``TIMESTAMPTZ`` and
    ``AT TIME ZONE``, the ``TIME`` type, ``INTERVAL`` values themselves
    (e.g. as a result column), and sub-second date parts.
    """

    SELECT_KINDS: tuple[str, ...] = ()
    TRY_SUPPORTED = False
    SUPPORTS_UESCAPE = False
    SUPPORTS_DECODE_CASE = False
    SET_OP_PARENTHESIZED_OPERANDS = False

    AFTER_HAVING_MODIFIER_TRANSFORMS = generator.AFTER_HAVING_MODIFIER_TRANSFORMS

    JOIN_HINTS = False
    TABLE_HINTS = False
    QUERY_HINTS = False
    NVL2_SUPPORTED = False
    JSON_PATH_BRACKETED_KEY_SUPPORTED = False
    SUPPORTS_CREATE_TABLE_LIKE = False
    SUPPORTS_TABLE_ALIAS_COLUMNS = False
    SUPPORTS_TO_NUMBER = False
    SUPPORTS_WINDOW_EXCLUDE = True
    EXCEPT_INTERSECT_SUPPORT_ALL_CLAUSE = False
    SUPPORTS_MEDIAN = False
    JSON_KEY_VALUE_PAIR_SEP = ","
    PARSE_JSON_NAME: str | None = None

    SUPPORTED_JSON_PATH_PARTS = {
        exp.JSONPathKey,
        exp.JSONPathRoot,
        exp.JSONPathSubscript,
    }

    TYPE_MAPPING = {
        **{k: v for k, v in generator.Generator.TYPE_MAPPING.items() if k != exp.DType.BLOB},
        exp.DType.BOOLEAN: "INTEGER",
        exp.DType.TINYINT: "INTEGER",
        exp.DType.SMALLINT: "INTEGER",
        exp.DType.INT: "INTEGER",
        exp.DType.BIGINT: "INTEGER",
        exp.DType.FLOAT: "REAL",
        exp.DType.DOUBLE: "REAL",
        exp.DType.DECIMAL: "REAL",
        exp.DType.CHAR: "TEXT",
        exp.DType.NCHAR: "TEXT",
        exp.DType.VARCHAR: "TEXT",
        exp.DType.NVARCHAR: "TEXT",
        exp.DType.BINARY: "BLOB",
        exp.DType.VARBINARY: "BLOB",
    }

    def nullsafeeq_sql(self, expression: exp.NullSafeEQ) -> str:
        return self.binary(expression, "IS")

    def nullsafeneq_sql(self, expression: exp.NullSafeNEQ) -> str:
        return self.binary(expression, "IS NOT")

    TOKEN_MAPPING = {
        TokenType.AUTO_INCREMENT: "AUTOINCREMENT",
    }

    TRANSFORMS = {
        **generator.Generator.TRANSFORMS,
        exp.AnyValue: any_value_to_max_sql,
        exp.Chr: rename_func("CHAR"),
        exp.CountIf: count_if_to_sum,
        exp.Create: transforms.preprocess([_transform_create]),
        exp.CurrentDate: lambda *_: "CURRENT_DATE",
        exp.CurrentTime: lambda *_: "CURRENT_TIME",
        exp.CurrentTimestamp: lambda *_: "CURRENT_TIMESTAMP",
        exp.CurrentVersion: lambda *_: "SQLITE_VERSION()",
        exp.ColumnDef: transforms.preprocess([_generated_to_auto_increment]),
        exp.DateStrToDate: lambda self, e: self.sql(e, "this"),
        exp.If: rename_func("IIF"),
        exp.ILike: no_ilike_sql,
        exp.JSONArrayAgg: unsupported_args("order", "null_handling", "return_type", "strict")(
            rename_func("JSON_GROUP_ARRAY")
        ),
        exp.JSONObjectAgg: lambda self, e: self._jsonobject_sql(e, name="JSON_GROUP_OBJECT"),
        exp.Levenshtein: unsupported_args("ins_cost", "del_cost", "sub_cost", "max_dist")(
            rename_func("EDITDIST3")
        ),
        exp.LogicalOr: rename_func("MAX"),
        exp.LogicalAnd: rename_func("MIN"),
        exp.Pivot: no_pivot_sql,
        exp.Rand: rename_func("RANDOM"),
        exp.RegexpLike: lambda self, e: self.binary(e, "REGEXP"),
        exp.Select: transforms.preprocess(
            [
                transforms.eliminate_distinct_on,
                transforms.eliminate_qualify,
                transforms.eliminate_semi_and_anti_joins,
            ]
        ),
        exp.StrPosition: lambda self, e: strposition_sql(self, e, func_name="INSTR"),
        exp.TableSample: no_tablesample_sql,
        exp.TimeStrToTime: lambda self, e: self.sql(e, "this"),
        exp.TimeToStr: lambda self, e: self.func("STRFTIME", e.args.get("format"), e.this),
        exp.TsOrDsToTimestamp: lambda self, e: self.sql(e, "this"),
        # DuckDB date-part synonym functions, e.g. YEAR(x) == EXTRACT(year FROM x)
        exp.Year: lambda self, e: self._date_part_sql("YEAR", self.sql(e, "this")),
        exp.Quarter: lambda self, e: self._date_part_sql("QUARTER", self.sql(e, "this")),
        exp.Month: lambda self, e: self._date_part_sql("MONTH", self.sql(e, "this")),
        exp.Day: lambda self, e: self._date_part_sql("DAY", self.sql(e, "this")),
        exp.DayOfWeek: lambda self, e: self._date_part_sql("DOW", self.sql(e, "this")),
        exp.DayOfYear: lambda self, e: self._date_part_sql("DOY", self.sql(e, "this")),
        exp.Week: lambda self, e: self._date_part_sql("WEEK", self.sql(e, "this")),
        exp.Hour: lambda self, e: self._date_part_sql("HOUR", self.sql(e, "this")),
        exp.Minute: lambda self, e: self._date_part_sql("MINUTE", self.sql(e, "this")),
        exp.Second: lambda self, e: self._date_part_sql("SECOND", self.sql(e, "this")),
    }

    # DuckDB type families used to classify temporal operands; see _temporal_kind.
    _TIMESTAMP_DTYPES = {
        exp.DType.TIMESTAMP,
        exp.DType.TIMESTAMPNTZ,
        exp.DType.DATETIME,
        exp.DType.TIMESTAMP_S,
        exp.DType.TIMESTAMP_MS,
        exp.DType.TIMESTAMP_NS,
    }
    _TEXT_DTYPES = {
        exp.DType.TEXT,
        exp.DType.VARCHAR,
        exp.DType.CHAR,
        exp.DType.NCHAR,
        exp.DType.NVARCHAR,
    }
    _NUMBER_DTYPES = {
        exp.DType.INT,
        exp.DType.BIGINT,
        exp.DType.SMALLINT,
        exp.DType.TINYINT,
        exp.DType.UINT,
        exp.DType.UBIGINT,
        exp.DType.USMALLINT,
        exp.DType.UTINYINT,
        exp.DType.FLOAT,
        exp.DType.DOUBLE,
        exp.DType.DECIMAL,
    }

    # INTERVAL units that map to a SQLite date/time modifier, and month multipliers
    # for the units that need clamping month arithmetic.
    _INTERVAL_MODIFIER_UNITS = {
        "DAY": "days",
        "HOUR": "hours",
        "MINUTE": "minutes",
        "SECOND": "seconds",
    }
    _INTERVAL_MONTH_MULTIPLIERS = {"YEAR": 12, "QUARTER": 3, "MONTH": 1}

    # EXTRACT/DATE_PART aliases to canonical DuckDB part names.
    _EXTRACT_PART_ALIASES = {
        "YEARS": "YEAR",
        "QUARTERS": "QUARTER",
        "MONTHS": "MONTH",
        "WEEKS": "WEEK",
        "DAYS": "DAY",
        "HOURS": "HOUR",
        "MINUTES": "MINUTE",
        "SECONDS": "SECOND",
        "DAYOFWEEK": "DOW",
        "WEEKDAY": "DOW",
        "DAYOFYEAR": "DOY",
    }

    # SQLite doesn't generally support CREATE TABLE .. properties
    # https://www.sqlite.org/lang_createtable.html
    PROPERTIES_LOCATION = {
        **{
            prop: exp.Properties.Location.UNSUPPORTED
            for prop in generator.Generator.PROPERTIES_LOCATION
        },
        # There are a few exceptions (e.g. temporary tables) which are supported or
        # can be transpiled to SQLite, so we explicitly override them accordingly
        exp.LikeProperty: exp.Properties.Location.POST_SCHEMA,
        exp.ModuleProperty: exp.Properties.Location.POST_SCHEMA,
        exp.StrictProperty: exp.Properties.Location.POST_SCHEMA,
        exp.TemporaryProperty: exp.Properties.Location.POST_CREATE,
        exp.VirtualProperty: exp.Properties.Location.POST_CREATE,
        exp.WithoutRowidProperty: exp.Properties.Location.POST_SCHEMA,
    }

    LIMIT_FETCH = "LIMIT"

    def root_properties(self, properties: exp.Properties) -> str:
        return self.expressions(properties, indent=False, sep=", ")

    def offset_limit_modifiers(
        self, expression: exp.Expr, fetch: bool, limit: exp.Fetch | exp.Limit | None
    ) -> list[str]:
        if expression.args.get("offset") and (
            not limit or (isinstance(limit, exp.Limit) and limit.is_limit_all)
        ):
            limit = exp.Limit(expression=exp.Literal.number(-1))
        return super().offset_limit_modifiers(expression, fetch, limit)

    def subquery_sql(self, expression: exp.Subquery, sep: str = " AS ") -> str:
        if any(expression.args.get(key) for key in ("order", "limit", "offset")):
            query = expression.this
            keys = ("order", "limit", "offset", "with_")
            # Merge grouping modifiers unless doing so would overwrite an inner clause.
            if any(expression.args.get(key) and query.args.get(key) for key in keys):
                query = exp.select("*").from_(query.subquery(copy=False), copy=False)
            for key in keys:
                if value := expression.args.get(key):
                    query.set(key, value.pop())
            return self.sql(query)

        return super().subquery_sql(expression, sep=sep)

    def autoincrementcolumnconstraint_sql(
        self, expression: exp.AutoIncrementColumnConstraint
    ) -> str:
        column = expression.find_ancestor(exp.ColumnDef)
        if column and column.find(exp.PrimaryKeyColumnConstraint):
            return super().autoincrementcolumnconstraint_sql(expression)

        self.unsupported("SQLite AUTOINCREMENT requires an INTEGER PRIMARY KEY")
        return ""

    def insert_sql(self, expression: exp.Insert) -> str:
        if expression.args.get("ignore"):
            expression.set("ignore", False)
            expression.set("alternative", "IGNORE")

        return super().insert_sql(expression)

    def bitwiseandagg_sql(self, expression: exp.BitwiseAndAgg) -> str:
        self.unsupported("BITWISE_AND aggregation is not supported in SQLite")
        return self.function_fallback_sql(expression)

    def bitwiseoragg_sql(self, expression: exp.BitwiseOrAgg) -> str:
        self.unsupported("BITWISE_OR aggregation is not supported in SQLite")
        return self.function_fallback_sql(expression)

    def bitwisexoragg_sql(self, expression: exp.BitwiseXorAgg) -> str:
        self.unsupported("BITWISE_XOR aggregation is not supported in SQLite")
        return self.function_fallback_sql(expression)

    def jsonextract_sql(self, expression: exp.JSONExtract) -> str:
        if expression.expressions:
            return self.function_fallback_sql(expression)
        return arrow_json_extract_sql(self, expression)

    def jsonextractscalar_sql(self, expression: exp.JSONExtractScalar) -> str:
        if expression.args.get("json_subtype"):
            # json_extract() keeps the JSON subtype on object/array results;
            # ->> strips it, observable when the result feeds another JSON function
            return self.func("JSON_EXTRACT", expression.this, expression.expression)
        return arrow_json_extract_sql(self, expression)

    def dateadd_sql(self, expression: exp.DateAdd) -> str:
        if expression.meta_get("duckdb"):
            # DuckDB's DATE_ADD(x, INTERVAL ...) is exactly x + INTERVAL ...,
            # i.e. the result is a TIMESTAMP even for DATE arguments.
            sql = self._interval_arithmetic_sql(
                exp.Add(this=expression.this, expression=expression.expression), subtract=False
            )
            if sql is not None:
                return sql

        modifier = expression.expression
        unit = expression.args.get("unit")
        # An INTERVAL amount carries its own unit, e.g. DATE_ADD(d, INTERVAL 1 DAY);
        # unwrap it so the unit is not left inside the quoted modifier string.
        if isinstance(modifier, exp.Interval):
            unit = unit or modifier.unit
            modifier = modifier.this

        modifier = modifier.unnest()
        if (
            unit
            and not isinstance(modifier, exp.Literal)
            and not (isinstance(modifier, exp.Neg) and modifier.this.is_number)
        ):
            return self.func(
                "DATE",
                expression.this,
                exp.DPipe(
                    this=exp.paren(modifier),
                    expression=exp.Literal.string(f" {unit.name}"),
                ),
            )
        modifier = modifier.name if modifier.is_string else self.sql(modifier)
        modifier = f"{modifier} {unit.name}" if unit else modifier
        return self.func("DATE", expression.this, exp.Literal.string(modifier))

    def datesub_sql(self, expression: exp.DateSub) -> str:
        # SQLite has no DATE_SUB, so add the negated amount instead
        amount = expression.expression
        unit = expression.args.get("unit")
        if isinstance(amount, exp.Interval):
            unit = unit or amount.unit
            amount = amount.this

        amount = amount.unnest()
        negated: exp.Expr
        if isinstance(amount, exp.Neg):
            negated = amount.this
        elif isinstance(amount, exp.Literal):
            text = amount.name.removeprefix("+")
            negated = exp.Literal.string(text[1:] if text.startswith("-") else f"-{text}")
        else:
            negated = exp.Neg(
                this=amount
                if isinstance(amount, (exp.Column, exp.Func))
                else exp.paren(amount, copy=False)
            )

        return self.dateadd_sql(exp.DateAdd(this=expression.this, expression=negated, unit=unit))

    def cast_sql(self, expression: exp.Cast, safe_prefix: str | None = None) -> str:
        dtype = expression.to.this if isinstance(expression.to, exp.DataType) else None

        if dtype == exp.DType.DATE:
            return self._cast_to_date_sql(expression)
        if dtype in self._TIMESTAMP_DTYPES:
            return self._cast_to_timestamp_sql(expression)
        if dtype in (exp.DType.TIMESTAMPTZ, exp.DType.TIMESTAMPLTZ):
            self.unsupported(
                "TIMESTAMPTZ is not supported in SQLite; generating a plain TIMESTAMP cast"
            )
            return self._cast_to_timestamp_sql(expression)
        if dtype == exp.DType.TIME:
            self.unsupported("TIME is not supported in SQLite")
            return self.func("TIME", expression.this)
        if dtype == exp.DType.INTERVAL:
            self.unsupported("INTERVAL values are not supported in SQLite")
            return super().cast_sql(expression, safe_prefix)

        return super().cast_sql(expression, safe_prefix)

    def trycast_sql(self, expression: exp.TryCast) -> str:
        # SQLite has no TRY_CAST, but the DATE/TIMESTAMP casts generated here return
        # NULL for invalid input, which matches DuckDB's TRY_CAST semantics exactly.
        return self.cast_sql(expression)

    def _cast_to_date_sql(self, expression: exp.Cast) -> str:
        """CAST/TRY_CAST(x AS DATE): DuckDB's DATE is 'YYYY-MM-DD' text in SQLite."""
        this = expression.this
        if isinstance(this, exp.Null):
            return "NULL"

        kind = self._temporal_kind(this)
        if kind in ("timestamptz", "time", "interval"):
            self.unsupported(f"CAST from {kind.upper()} to DATE is not supported in SQLite")
            return self.func("DATE", this)

        if kind == "text":
            literal = self._string_literal_text(this)
            if literal is not None:
                normalized = _normalize_temporal_literal(literal, "date")
                if normalized is not None:
                    # Keep the original text if SQLite's DATE() already accepts it
                    if _SQLITE_DATE_TEXT_RE.match(literal.strip()):
                        return self.func("DATE", this)
                    return self.func("DATE", exp.Literal.string(normalized))
                if isinstance(expression, exp.TryCast):
                    return "NULL"
                # DuckDB's CAST raises on invalid dates; SQLite's DATE() yields NULL,
                # which is acceptable (the query has no result in DuckDB anyway).
                return self.func("DATE", this)

        if kind in (None, "text"):
            # Runtime-validate so invalid strings yield NULL, like DuckDB's TRY_CAST.
            return (
                "(SELECT CASE WHEN __d GLOB '*-*-*' AND "
                "DATE(JULIANDAY(DATE(__d))) = DATE(__d) THEN DATE(__d) END FROM "
                + self._normalized_datetime_subquery_sql(self.sql(this))
                + ")"
            )

        return self.func("DATE", this)

    def _cast_to_timestamp_sql(self, expression: exp.Cast) -> str:
        """CAST/TRY_CAST(x AS TIMESTAMP): DuckDB's TIMESTAMP is 'YYYY-MM-DD HH:MM:SS' text."""
        this = expression.this
        if isinstance(this, exp.Null):
            return "NULL"

        kind = self._temporal_kind(this)
        if kind in ("time", "interval"):
            self.unsupported(
                f"CAST from {kind.upper()} to TIMESTAMP is not supported in SQLite"
            )
            return self.func("DATETIME", this)

        if kind == "text":
            literal = self._string_literal_text(this)
            if literal is not None:
                normalized = _normalize_temporal_literal(literal, "timestamp")
                if normalized is not None:
                    canonical, fraction = normalized
                    # Keep the original text if SQLite's DATETIME() already accepts it
                    if _SQLITE_DATE_TEXT_RE.match(literal.strip()):
                        sql = self.func("DATETIME", this)
                    else:
                        sql = self.func("DATETIME", exp.Literal.string(canonical))
                    if fraction:
                        sql = f"{sql} || {self.sql(exp.Literal.string(fraction))}"
                    return sql
                if isinstance(expression, exp.TryCast):
                    return "NULL"
                # DuckDB's CAST raises; SQLite's DATETIME() yields NULL (acceptable).
                return self.func("DATETIME", this)

        if kind in (None, "text"):
            return (
                "(SELECT CASE WHEN __d GLOB '*-*-*' AND "
                "DATE(JULIANDAY(DATE(__d))) = DATE(__d) "
                "THEN DATETIME(JULIANDAY(__d)) || "
                "CASE WHEN INSTR(__d, '.') > 0 THEN SUBSTR(__d, INSTR(__d, '.')) ELSE '' END "
                "END FROM "
                + self._normalized_datetime_subquery_sql(self.sql(this))
                + ")"
            )

        if kind == "timestamp":
            # DATETIME() alone would drop fractional seconds
            return self._append_fraction_sql(self.func("DATETIME", this), this)

        return self.func("DATETIME", this)

    def _fraction_suffix_sql(self, x: str) -> str:
        """SQL fragment evaluating to x's fractional-seconds suffix ('.123') or ''."""
        return f"CASE WHEN INSTR({x}, '.') > 0 THEN SUBSTR({x}, INSTR({x}, '.')) ELSE '' END"

    def _append_fraction_sql(self, sql: str, x: exp.Expr) -> str:
        return f"{sql} || {self._fraction_suffix_sql(self.sql(x))}"

    def _normalized_datetime_text_sql(self, s: str) -> str:
        """SQL expression normalizing DuckDB-lenient datetime text to the strict
        'YYYY-MM-DD[ HH:MM:SS[.fff]]' form that SQLite's date/time functions accept:
        trims whitespace, maps '/' and 'T' separators, and zero-pads components."""
        b = f"REPLACE(REPLACE(TRIM({s}), 'T', ' '), '/', '-')"
        pos = f"INSTR({b} || ' ', ' ')"
        date_part = f"SUBSTR({b}, 1, {pos} - 1)"
        time_part = f"SUBSTR({b}, {pos} + 1)"

        padded_date = (
            f"CASE"
            f" WHEN {date_part} GLOB '????-?-?' THEN SUBSTR({date_part}, 1, 5) || '0' || SUBSTR({date_part}, 6, 2) || '0' || SUBSTR({date_part}, 8)"
            f" WHEN {date_part} GLOB '????-??-?' THEN SUBSTR({date_part}, 1, 8) || '0' || SUBSTR({date_part}, 9)"
            f" WHEN {date_part} GLOB '????-?-??' THEN SUBSTR({date_part}, 1, 5) || '0' || SUBSTR({date_part}, 6)"
            f" ELSE {date_part} END"
        )
        padded_time = (
            f"CASE"
            f" WHEN {time_part} GLOB '?:?:?' OR {time_part} GLOB '?:?:?.*' THEN '0' || SUBSTR({time_part}, 1, 1) || ':0' || SUBSTR({time_part}, 3, 1) || ':0' || SUBSTR({time_part}, 5)"
            f" WHEN {time_part} GLOB '?:?:??' OR {time_part} GLOB '?:?:??.*' THEN '0' || SUBSTR({time_part}, 1, 1) || ':0' || SUBSTR({time_part}, 3, 1) || ':' || SUBSTR({time_part}, 5)"
            f" WHEN {time_part} GLOB '?:??:?' OR {time_part} GLOB '?:??:?.*' THEN '0' || SUBSTR({time_part}, 1, 1) || ':' || SUBSTR({time_part}, 3, 2) || ':0' || SUBSTR({time_part}, 6)"
            f" WHEN {time_part} GLOB '?:??:??' OR {time_part} GLOB '?:??:??.*' THEN '0' || SUBSTR({time_part}, 1, 1) || ':' || SUBSTR({time_part}, 3, 2) || ':' || SUBSTR({time_part}, 6)"
            f" WHEN {time_part} GLOB '??:?:?' OR {time_part} GLOB '??:?:?.*' THEN SUBSTR({time_part}, 1, 2) || ':0' || SUBSTR({time_part}, 4, 1) || ':0' || SUBSTR({time_part}, 6)"
            f" WHEN {time_part} GLOB '??:?:??' OR {time_part} GLOB '??:?:??.*' THEN SUBSTR({time_part}, 1, 2) || ':0' || SUBSTR({time_part}, 4, 1) || ':' || SUBSTR({time_part}, 6)"
            f" WHEN {time_part} GLOB '??:??:?' OR {time_part} GLOB '??:??:?.*' THEN SUBSTR({time_part}, 1, 2) || ':' || SUBSTR({time_part}, 4, 2) || ':0' || SUBSTR({time_part}, 7)"
            f" WHEN {time_part} GLOB '?:?' THEN '0' || SUBSTR({time_part}, 1, 1) || ':0' || SUBSTR({time_part}, 3)"
            f" WHEN {time_part} GLOB '?:??' THEN '0' || SUBSTR({time_part}, 1, 1) || ':' || SUBSTR({time_part}, 3)"
            f" WHEN {time_part} GLOB '??:?' THEN SUBSTR({time_part}, 1, 2) || ':0' || SUBSTR({time_part}, 4)"
            f" ELSE {time_part} END"
        )
        return (
            f"CASE WHEN {time_part} = '' THEN {padded_date} "
            f"ELSE {padded_date} || ' ' || {padded_time} END"
        )

    def _temporal_kind(self, e: exp.Expr) -> str | None:
        """Classify an expression's DuckDB type family for temporal transpilation.

        Uses type annotations (e.g. from sqlglot.optimizer.annotate_types) when
        available and falls back to literal/CAST syntax. Returns one of "date",
        "timestamp", "timestamptz", "time", "text", "number", "interval", "null",
        or None when the type cannot be determined.
        """
        while isinstance(e, exp.Paren):
            e = e.this

        if isinstance(e, exp.Interval):
            return "interval"
        if isinstance(e, exp.Null):
            return "null"

        dtype = e.type
        if isinstance(dtype, exp.DataType):
            dtype = dtype.this
        elif isinstance(e, (exp.Cast, exp.TryCast)) and isinstance(e.to, exp.DataType):
            dtype = e.to.this
        else:
            dtype = None

        if dtype is not None:
            if dtype == exp.DType.DATE:
                return "date"
            if dtype in self._TIMESTAMP_DTYPES:
                return "timestamp"
            if dtype in (exp.DType.TIMESTAMPTZ, exp.DType.TIMESTAMPLTZ):
                return "timestamptz"
            if dtype == exp.DType.TIME:
                return "time"
            if dtype in self._TEXT_DTYPES:
                return "text"
            if dtype in self._NUMBER_DTYPES:
                return "number"
            if dtype == exp.DType.INTERVAL:
                return "interval"
            return None

        if isinstance(e, exp.Literal):
            return "text" if e.is_string else "number"
        if isinstance(e, exp.CurrentDate):
            return "date"
        if isinstance(e, exp.CurrentTimestamp):
            return "timestamp"
        if isinstance(e, (exp.DateTrunc, exp.TimestampTrunc, exp.TsOrDsToTimestamp)):
            return "timestamp"
        if isinstance(e, (exp.LastDay, exp.TsOrDsToDate, exp.Date)):
            return "date"
        return None

    @staticmethod
    def _string_literal_text(e: exp.Expr) -> str | None:
        e = e.unnest()
        return e.this if isinstance(e, exp.Literal) and e.is_string else None

    def _interval_amount_sql(self, amount: exp.Expr) -> tuple[int | None, str | None]:
        """Split an INTERVAL amount into a Python int (for signed integer literals)
        or a SQL expression string (anything else)."""
        amount = amount.unnest()
        if isinstance(amount, exp.Literal):
            text = amount.this
            if not amount.is_string or re.fullmatch(r"[+-]?\d+", text.strip()):
                return int(text), None
        elif isinstance(amount, exp.Neg) and isinstance(amount.this, exp.Literal):
            literal = amount.this
            if not literal.is_string or re.fullmatch(r"[+-]?\d+", literal.this.strip()):
                return -int(literal.this), None
        return None, self.sql(amount)

    def _interval_arithmetic_sql(self, expression: exp.Binary, subtract: bool) -> str | None:
        """SQL for `<temporal> +/- INTERVAL n unit` (DuckDB semantics: the result is
        always a TIMESTAMP, i.e. 'YYYY-MM-DD HH:MM:SS' text). Returns None when no
        INTERVAL operand is involved (or when falling back to the default rendering
        after reporting an unsupported construct)."""
        this, expr = expression.this, expression.expression
        left_interval = isinstance(this.unnest(), exp.Interval)
        right_interval = isinstance(expr.unnest(), exp.Interval)

        if not (left_interval or right_interval):
            return None
        if left_interval and right_interval:
            # INTERVAL +/- INTERVAL yields an INTERVAL, which SQLite cannot represent
            return None
        if left_interval:
            if subtract:
                # INTERVAL - x is not valid in DuckDB
                return None
            interval, operand, negated = this.unnest(), expr, False
        else:
            interval, operand, negated = expr.unnest(), this, subtract

        unit = interval.args.get("unit")
        amount = interval.this
        if unit is None or amount is None:
            self.unsupported("INTERVAL without a unit is not supported in SQLite")
            return None

        kind = self._temporal_kind(operand)
        if kind in ("timestamptz", "time"):
            self.unsupported(
                f"INTERVAL arithmetic on {kind.upper()} is not supported in SQLite"
            )
            return None

        unit_name = unit.name.upper()
        if unit_name.endswith("S") and unit_name[:-1] in (
            set(self._INTERVAL_MODIFIER_UNITS) | set(self._INTERVAL_MONTH_MULTIPLIERS) | {"WEEK"}
        ):
            unit_name = unit_name[:-1]

        x = self.sql(operand)
        assume_date = kind == "date"

        if unit_name in self._INTERVAL_MONTH_MULTIPLIERS:
            return self._add_months_sql(
                x, amount, self._INTERVAL_MONTH_MULTIPLIERS[unit_name], negated, assume_date
            )
        if unit_name == "WEEK":
            return self._add_weeks_sql(x, amount, negated, assume_date)
        if unit_name in self._INTERVAL_MODIFIER_UNITS:
            return self._add_modifier_sql(
                x, amount, self._INTERVAL_MODIFIER_UNITS[unit_name], negated, assume_date
            )

        self.unsupported(f"INTERVAL unit '{unit.name}' is not supported in SQLite")
        return None

    def _add_months_sql(
        self, x: str, amount: exp.Expr, multiplier: int, negated: bool, assume_date: bool
    ) -> str:
        """`x +/- INTERVAL amount <month-like unit>` with DuckDB's end-of-month
        clamping (2024-01-31 + 1 month = 2024-02-29). SQLite's own 'months'
        modifier overflows into the next month instead, so the target month is
        computed from its (always valid) first day."""
        value, amount_sql = self._interval_amount_sql(amount)
        if value is not None:
            months = value * multiplier
            if negated:
                months = -months
            modifier = self.sql(exp.Literal.string(f"{months:+d} months"))
            guard = None
        else:
            k = f"CAST(({amount_sql}) AS INTEGER)"
            if multiplier != 1:
                k = f"{multiplier} * {k}"
            if negated:
                k = f"-({k})"
            modifier = f"PRINTF('%+d months', {k})"
            guard = k

        # First day of the target month: month arithmetic on day 1 never overflows.
        month_start = f"DATE(STRFTIME('%Y-%m-01', {x}), {modifier})"
        last_day = f"CAST(STRFTIME('%d', DATE({month_start}, '+1 month', '-1 day')) AS INTEGER)"
        day = f"MIN(CAST(STRFTIME('%d', {x}) AS INTEGER), {last_day})"
        if assume_date:
            time = "' 00:00:00'"
        else:
            time = f"CASE WHEN LENGTH({x}) > 10 THEN SUBSTR({x}, 11) ELSE ' 00:00:00' END"
        result = f"STRFTIME('%Y-%m-', {month_start}) || PRINTF('%02d', {day}) || {time}"
        if guard:
            result = f"CASE WHEN {guard} IS NULL THEN NULL ELSE {result} END"
        return result

    def _add_weeks_sql(self, x: str, amount: exp.Expr, negated: bool, assume_date: bool) -> str:
        """`x +/- INTERVAL amount WEEK` (a week is exactly 7 days)."""
        value, amount_sql = self._interval_amount_sql(amount)
        if value is not None:
            days = 7 * value
            if negated:
                days = -days
            modifier = self.sql(exp.Literal.string(f"{days:+d} days"))
            guard = None
        else:
            k = f"7 * CAST(({amount_sql}) AS INTEGER)"
            if negated:
                k = f"-({k})"
            modifier = f"PRINTF('%+d days', {k})"
            guard = k
        return self._apply_modifier_sql(x, modifier, guard, assume_date)

    def _add_modifier_sql(
        self, x: str, amount: exp.Expr, unit_plural: str, negated: bool, assume_date: bool
    ) -> str:
        """`x +/- INTERVAL amount <day/hour/minute/second>` via a SQLite modifier."""
        value, amount_sql = self._interval_amount_sql(amount)
        if value is not None:
            if negated:
                value = -value
            modifier = self.sql(exp.Literal.string(f"{value:+d} {unit_plural}"))
            guard = None
        else:
            if negated:
                amount_sql = f"-({amount_sql})"
            modifier = f"PRINTF('%+d {unit_plural}', {amount_sql})"
            guard = amount_sql
        return self._apply_modifier_sql(x, modifier, guard, assume_date)

    def _apply_modifier_sql(
        self, x: str, modifier: str, guard: str | None, assume_date: bool
    ) -> str:
        """DATETIME(x, modifier), preserving fractional seconds (which DATETIME()
        alone drops) and propagating a NULL interval amount."""
        result = f"DATETIME({x}, {modifier})"
        if not assume_date:
            result = f"{result} || {self._fraction_suffix_sql(x)}"
        if guard:
            result = f"CASE WHEN {guard} IS NULL THEN NULL ELSE {result} END"
        return result

    def _add_days_sql(self, date: exp.Expr, amount: exp.Expr, negated: bool) -> str:
        """`<date> +/- <integer>`: DuckDB adds whole days and the result stays a DATE."""
        value, amount_sql = self._interval_amount_sql(amount)
        if value is not None:
            if negated:
                value = -value
            return self.func("DATE", date, exp.Literal.string(f"{value:+d} days"))
        if negated:
            amount_sql = f"-({amount_sql})"
        result = f"DATE({self.sql(date)}, PRINTF('%+d days', {amount_sql}))"
        return f"CASE WHEN {amount_sql} IS NULL THEN NULL ELSE {result} END"

    def _temporal_add_sub_sql(self, expression: exp.Binary, subtract: bool) -> str | None:
        sql = self._interval_arithmetic_sql(expression, subtract)
        if sql is not None:
            return sql

        this, expr = expression.this, expression.expression
        left_kind = self._temporal_kind(this)
        right_kind = self._temporal_kind(expr)

        if subtract:
            if left_kind == "date" and right_kind == "number":
                return self._add_days_sql(this, expr, negated=True)
            if left_kind == "date" and right_kind == "date":
                # DuckDB's DATE - DATE yields the number of days (BIGINT)
                return (
                    f"CAST((JULIANDAY({self.sql(this)}) - JULIANDAY({self.sql(expr)})) "
                    f"AS INTEGER)"
                )
            if left_kind in ("date", "timestamp") and right_kind in ("date", "timestamp"):
                self.unsupported(
                    "Subtracting two temporal values yields an INTERVAL, which SQLite does not support"
                )
                return None
        else:
            if left_kind == "date" and right_kind == "number":
                return self._add_days_sql(this, expr, negated=False)
            if left_kind == "number" and right_kind == "date":
                return self._add_days_sql(expr, this, negated=False)

        if (left_kind in ("date", "timestamp") and right_kind is None) or (
            right_kind in ("date", "timestamp") and left_kind is None
        ):
            self.unsupported(
                "Cannot determine the operand types of the temporal "
                + ("subtraction" if subtract else "addition")
                + " for SQLite"
            )
        return None

    def add_sql(self, expression: exp.Add) -> str:
        sql = self._temporal_add_sub_sql(expression, subtract=False)
        return sql if sql is not None else super().add_sql(expression)

    def sub_sql(self, expression: exp.Sub) -> str:
        sql = self._temporal_add_sub_sql(expression, subtract=True)
        return sql if sql is not None else super().sub_sql(expression)

    def interval_sql(self, expression: exp.Interval) -> str:
        self.unsupported(
            "INTERVAL values are not supported in SQLite outside of date/time arithmetic"
        )
        return super().interval_sql(expression)

    def attimezone_sql(self, expression: exp.AtTimeZone) -> str:
        self.unsupported("AT TIME ZONE is not supported in SQLite")
        return super().attimezone_sql(expression)

    def _week_start_sql(self, x: str) -> str:
        """SQL expression for the Monday of x's ISO week (as DATE text)."""
        return f"DATE({x}, PRINTF('-%d days', (CAST(STRFTIME('%w', {x}) AS INTEGER) + 6) % 7))"

    def _iso_thursday_sql(self, x: str) -> str:
        """SQL expression for the Thursday of x's ISO week (the ISO week/year of a
        date is the week/year of its Thursday)."""
        isodow = f"((CAST(STRFTIME('%w', {x}) AS INTEGER) + 6) % 7 + 1)"
        return f"DATE({x}, PRINTF('%+d days', 4 - {isodow}))"

    def _date_part_sql(self, name: str, x: str) -> str | None:
        """SQL for EXTRACT(<name> FROM x) / DATE_PART('<name>', x), DuckDB semantics."""
        if name == "YEAR":
            return f"CAST(STRFTIME('%Y', {x}) AS INTEGER)"
        if name == "QUARTER":
            return f"((CAST(STRFTIME('%m', {x}) AS INTEGER) + 2) / 3)"
        if name == "MONTH":
            return f"CAST(STRFTIME('%m', {x}) AS INTEGER)"
        if name == "DAY":
            return f"CAST(STRFTIME('%d', {x}) AS INTEGER)"
        if name == "DOW":
            # DuckDB: Sunday = 0 .. Saturday = 6, same as SQLite's %w
            return f"CAST(STRFTIME('%w', {x}) AS INTEGER)"
        if name == "ISODOW":
            return f"((CAST(STRFTIME('%w', {x}) AS INTEGER) + 6) % 7 + 1)"
        if name == "DOY":
            return f"CAST(STRFTIME('%j', {x}) AS INTEGER)"
        if name == "WEEK":
            return f"((CAST(STRFTIME('%j', {self._iso_thursday_sql(x)}) AS INTEGER) + 6) / 7)"
        if name == "ISOYEAR":
            return f"CAST(STRFTIME('%Y', {self._iso_thursday_sql(x)}) AS INTEGER)"
        if name == "HOUR":
            return f"CAST(STRFTIME('%H', {x}) AS INTEGER)"
        if name == "MINUTE":
            return f"CAST(STRFTIME('%M', {x}) AS INTEGER)"
        if name == "SECOND":
            return f"CAST(STRFTIME('%S', {x}) AS INTEGER)"
        if name == "EPOCH":
            # DuckDB returns a DOUBLE; computed as microseconds to stay exact for
            # fractional-second input
            fraction = (
                f"CASE WHEN INSTR({x}, '.') > 0 "
                f"THEN CAST(SUBSTR({x}, INSTR({x}, '.')) AS REAL) ELSE 0.0 END"
            )
            return (
                f"((UNIXEPOCH({x}) * 1000000 + CAST(ROUND({fraction} * 1000000) AS INTEGER)) "
                f"/ 1000000.0)"
            )
        return None

    def extract_sql(self, expression: exp.Extract) -> str:
        part = expression.this
        name = part.name.upper() if part else ""
        name = self._EXTRACT_PART_ALIASES.get(name, name)
        sql = self._date_part_sql(name, self.sql(expression, "expression"))
        if sql is None:
            self.unsupported(f"EXTRACT({name} ...) is not supported in SQLite")
            return super().extract_sql(expression)
        return sql

    def datetrunc_sql(self, expression: exp.DateTrunc) -> str:
        return self._date_trunc_sql(expression)

    def timestamptrunc_sql(self, expression: exp.TimestampTrunc) -> str:
        return self._date_trunc_sql(expression)

    def _date_trunc_sql(self, expression: exp.DateTrunc | exp.TimestampTrunc) -> str:
        """DATE_TRUNC(unit, x): DuckDB always returns a TIMESTAMP."""
        unit = expression.args.get("unit")
        unit_name = unit.name.upper() if unit and unit.name else ""
        if unit_name.endswith("S") and unit_name[:-1] in (
            "YEAR", "QUARTER", "MONTH", "WEEK", "DAY", "HOUR", "MINUTE", "SECOND"
        ):
            unit_name = unit_name[:-1]

        x = self.sql(expression, "this")
        if unit_name == "YEAR":
            return f"STRFTIME('%Y-01-01 00:00:00', {x})"
        if unit_name == "QUARTER":
            return (
                f"STRFTIME('%Y-', {x}) || "
                f"PRINTF('%02d', ((CAST(STRFTIME('%m', {x}) AS INTEGER) - 1) / 3) * 3 + 1) || "
                f"'-01 00:00:00'"
            )
        if unit_name == "MONTH":
            return f"STRFTIME('%Y-%m-01 00:00:00', {x})"
        if unit_name == "WEEK":
            return f"DATETIME({self._week_start_sql(x)})"
        if unit_name == "DAY":
            return f"STRFTIME('%Y-%m-%d 00:00:00', {x})"
        if unit_name == "HOUR":
            return f"STRFTIME('%Y-%m-%d %H:00:00', {x})"
        if unit_name == "MINUTE":
            return f"STRFTIME('%Y-%m-%d %H:%M:00', {x})"
        if unit_name == "SECOND":
            return f"STRFTIME('%Y-%m-%d %H:%M:%S', {x})"

        self.unsupported(
            f"DATE_TRUNC with unit '{unit.name if unit else ''}' is not supported in SQLite"
        )
        return self.function_fallback_sql(expression)

    def lastday_sql(self, expression: exp.LastDay) -> str:
        unit = expression.args.get("unit")
        if unit and unit.name.upper() != "MONTH":
            self.unsupported("Date parts are not supported in LAST_DAY.")
        # DuckDB's LAST_DAY always returns a DATE
        return self.func(
            "DATE",
            expression.this,
            exp.Literal.string("start of month"),
            exp.Literal.string("+1 month"),
            exp.Literal.string("-1 day"),
        )

    def datediff_sql(self, expression: exp.DateDiff) -> str:
        """DATE_DIFF(unit, a, b): unit boundaries crossed between a and b (b - a),
        except WEEK which counts whole 7-day spans (DuckDB semantics)."""
        unit = expression.args.get("unit")
        unit_name = unit.name.upper() if unit and unit.name else "DAY"

        end = self.sql(expression, "this")
        start = self.sql(expression, "expression")

        def year(x: str) -> str:
            return f"CAST(STRFTIME('%Y', {x}) AS INTEGER)"

        def month(x: str) -> str:
            return f"CAST(STRFTIME('%m', {x}) AS INTEGER)"

        if unit_name == "DAY":
            return f"CAST((JULIANDAY(DATE({end})) - JULIANDAY(DATE({start}))) AS INTEGER)"
        if unit_name == "WEEK":
            boundary = expression.args.get("date_part_boundary")
            if boundary is False:
                # DuckDB: whole 7-day spans between the two calendar dates
                return f"CAST((JULIANDAY(DATE({end})) - JULIANDAY(DATE({start}))) / 7 AS INTEGER)"
            if boundary is None:
                self.unsupported("DATEDIFF unsupported for 'WEEK'.")
                return f"CAST((JULIANDAY({end}) - JULIANDAY({start})) AS INTEGER)"
            # ISO-week (Monday) boundaries
            return (
                f"CAST((JULIANDAY({self._week_start_sql(end)}) - "
                f"JULIANDAY({self._week_start_sql(start)})) / 7 AS INTEGER)"
            )
        if unit_name == "MONTH":
            return (
                f"(({year(end)} * 12 + {month(end)}) - ({year(start)} * 12 + {month(start)}))"
            )
        if unit_name == "QUARTER":
            return (
                f"(({year(end)} * 4 + ({month(end)} - 1) / 3) - "
                f"({year(start)} * 4 + ({month(start)} - 1) / 3))"
            )
        if unit_name == "YEAR":
            return f"({year(end)} - {year(start)})"
        if unit_name in ("HOUR", "MINUTE"):
            fmt = "%Y-%m-%d %H:00:00" if unit_name == "HOUR" else "%Y-%m-%d %H:%M:00"
            factor = 24 if unit_name == "HOUR" else 1440
            return (
                f"CAST(ROUND((JULIANDAY(STRFTIME('{fmt}', {end})) - "
                f"JULIANDAY(STRFTIME('{fmt}', {start}))) * {factor}) AS INTEGER)"
            )
        if unit_name == "SECOND":
            return f"(UNIXEPOCH({end}) - UNIXEPOCH({start}))"
        if unit_name == "MILLISECOND":
            return f"CAST((JULIANDAY({end}) - JULIANDAY({start})) * 86400000.0 AS INTEGER)"
        if unit_name == "MICROSECOND":
            return f"CAST((JULIANDAY({end}) - JULIANDAY({start})) * 86400000000.0 AS INTEGER)"
        if unit_name == "NANOSECOND":
            return f"CAST((JULIANDAY({end}) - JULIANDAY({start})) * 86400000000000.0 AS INTEGER)"

        self.unsupported(f"DATEDIFF unsupported for '{unit_name}'.")
        return f"CAST((JULIANDAY({end}) - JULIANDAY({start})) AS INTEGER)"

    def _normalize_temporal_comparison(self, expression: exp.Binary) -> None:
        """Adjust DATE-vs-TIMESTAMP comparisons in place (the generator works on a
        copy of the tree) so both sides share the same text representation."""
        this, expr = expression.this, expression.expression
        if this is None or expr is None:
            return

        left_kind = self._temporal_kind(this)
        right_kind = self._temporal_kind(expr)
        kinds = {left_kind, right_kind}

        if "timestamptz" in kinds or "time" in kinds:
            self.unsupported("TIMESTAMPTZ/TIME comparisons are not supported in SQLite")
            return
        if "date" not in kinds and "timestamp" not in kinds:
            return
        if left_kind is None or right_kind is None:
            self.unsupported(
                "Cannot determine both operand types of the temporal comparison for SQLite"
            )
            return
        if left_kind != right_kind and "text" not in kinds and "null" not in kinds:
            # DATE vs TIMESTAMP: DuckDB compares them as timestamps
            if left_kind == "date":
                expression.set("this", exp.func("DATETIME", this))
            if right_kind == "date":
                expression.set("expression", exp.func("DATETIME", expr))

    def eq_sql(self, expression: exp.EQ) -> str:
        self._normalize_temporal_comparison(expression)
        return super().eq_sql(expression)

    def neq_sql(self, expression: exp.NEQ) -> str:
        self._normalize_temporal_comparison(expression)
        return super().neq_sql(expression)

    def lt_sql(self, expression: exp.LT) -> str:
        self._normalize_temporal_comparison(expression)
        return super().lt_sql(expression)

    def lte_sql(self, expression: exp.LTE) -> str:
        self._normalize_temporal_comparison(expression)
        return super().lte_sql(expression)

    def gt_sql(self, expression: exp.GT) -> str:
        self._normalize_temporal_comparison(expression)
        return super().gt_sql(expression)

    def gte_sql(self, expression: exp.GTE) -> str:
        self._normalize_temporal_comparison(expression)
        return super().gte_sql(expression)

    def between_sql(self, expression: exp.Between) -> str:
        operands = [expression.this, expression.args.get("low"), expression.args.get("high")]
        kinds = [self._temporal_kind(e) for e in operands if e is not None]

        if "timestamptz" in kinds or "time" in kinds:
            self.unsupported("TIMESTAMPTZ/TIME comparisons are not supported in SQLite")
        elif "date" in kinds or "timestamp" in kinds:
            if len(kinds) < 3 or None in kinds:
                self.unsupported(
                    "Cannot determine all operand types of the temporal BETWEEN for SQLite"
                )
            elif "date" in kinds and "timestamp" in kinds:
                # DATE vs TIMESTAMP: DuckDB compares them as timestamps
                for key in ("this", "low", "high"):
                    operand = expression.args.get(key)
                    if operand is not None and self._temporal_kind(operand) == "date":
                        expression.set(key, exp.func("DATETIME", operand))
        return super().between_sql(expression)

    # https://www.sqlite.org/gencol.html
    # Inline unsupported check: mypyc cannot compile @unsupported_args on an
    # override of an undecorated base-class method.
    def computedcolumnconstraint_sql(self, expression: exp.ComputedColumnConstraint) -> str:
        if expression.args.get("data_type"):
            self.unsupported("SQLite generated columns do not support a data type")

        this = expression.this
        this_sql = self.sql(this) if isinstance(this, exp.Paren) else f"({self.sql(this)})"
        storage = " STORED" if expression.args.get("persisted") else ""
        not_null = " NOT NULL" if expression.args.get("not_null") else ""
        return f"AS {this_sql}{storage}{not_null}"

    # Note: SQLite's TRUNC always returns REAL (e.g., trunc(10.99) -> 10.0), not INTEGER.
    # This creates a transpilation gap affecting division semantics, similar to Presto.
    # Unlike Presto where this only affects decimals=0, SQLite has no decimals parameter
    # so every use of TRUNC is affected. Modeling precisely would require exp.FloatTrunc.
    @unsupported_args("decimals")
    def trunc_sql(self, expression: exp.Trunc) -> str:
        return self.func("TRUNC", expression.this)

    def generateseries_sql(self, expression: exp.GenerateSeries) -> str:
        parent = expression.parent
        alias = parent and parent.args.get("alias")

        if isinstance(alias, exp.TableAlias) and alias.columns:
            column_alias = alias.columns[0]
            alias.set("columns", None)
            sql = self.sql(
                exp.select(exp.alias_("value", column_alias)).from_(expression).subquery()
            )
        else:
            sql = self.function_fallback_sql(expression)

        return sql

    def groupconcat_sql(self, expression: exp.GroupConcat) -> str:
        node = expression.parent if isinstance(expression.parent, exp.Filter) else expression
        window = node.parent

        if (
            isinstance(expression.this, exp.Order)
            and isinstance(window, exp.Window)
            and window.this is node
        ):
            self.unsupported(
                "SQLite GROUP_CONCAT window functions do not support argument ORDER BY"
            )
            expression.set("this", expression.this.this)

        return groupconcat_sql(
            self, expression, func_name="GROUP_CONCAT", sep=None, within_group=False
        )

    def _greatest_least_sql(self, expression: exp.Greatest | exp.Least) -> str:
        if not expression.expressions:
            return self.sql(expression, "this")

        name = "MAX" if isinstance(expression, exp.Greatest) else "MIN"

        if not expression.args.get("ignore_nulls"):
            return rename_func(name)(self, expression)

        # SQLite's multi-argument MAX/MIN return NULL if any argument is NULL.
        # GREATEST(a, b, c) -> MAX(COALESCE(a, b, c), COALESCE(b, c, a), COALESCE(c, a, b)).
        args = [expression.this, *expression.expressions]

        coalesces = []
        for i in range(len(args)):
            rotated = args[i:] + args[:i]
            coalesces.append(exp.Coalesce(this=rotated[0], expressions=rotated[1:]))

        return self.func(name, *coalesces)

    def greatest_sql(self, expression: exp.Greatest) -> str:
        return self._greatest_least_sql(expression)

    def least_sql(self, expression: exp.Least) -> str:
        return self._greatest_least_sql(expression)

    def transaction_sql(self, expression: exp.Transaction) -> str:
        this = expression.this
        this = f" {this}" if this else ""
        return f"BEGIN{this} TRANSACTION"

    def isascii_sql(self, expression: exp.IsAscii) -> str:
        return f"(NOT {self.sql(expression.this)} GLOB CAST(x'2a5b5e012d7f5d2a' AS TEXT))"

    @unsupported_args("this")
    def currentschema_sql(self, expression: exp.CurrentSchema) -> str:
        return "'main'"

    def ignorenulls_sql(self, expression: exp.IgnoreNulls) -> str:
        self.unsupported("SQLite does not support IGNORE NULLS.")
        return self.sql(expression.this)

    def respectnulls_sql(self, expression: exp.RespectNulls) -> str:
        return self.sql(expression.this)

    def windowspec_sql(self, expression: exp.WindowSpec) -> str:
        if (
            expression.text("kind").upper() == "RANGE"
            and expression.text("start").upper() == "CURRENT ROW"
            and expression.args.get("end") is None
        ):
            return "RANGE CURRENT ROW"

        return super().windowspec_sql(expression)
