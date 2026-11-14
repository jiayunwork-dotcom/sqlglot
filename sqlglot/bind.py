"""
Parameter-aware transpilation.

``transpile_bound`` is the parameter-aware sibling of :func:`sqlglot.transpile`.
It parses one SQL statement together with the parameters it was written against,
transpiles the statement, and reports which source parameter ended up at each
parameter slot of the generated SQL. This lets callers hand the generated SQL and
the matching values to a database driver without re-deriving the binding order.

The module never stringifies, compares, hashes, or copies parameter values. Values
are looked up purely by their source identity (the lexical position of a ``?``,
the number of a ``$n`` marker, or the name of a ``:name`` marker) and passed
through unchanged.
"""

from __future__ import annotations

import re
import secrets
import typing as t
from collections.abc import Mapping

from sqlglot import expressions as exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import BindError, ParseError
from sqlglot.tokens import Token, TokenType

if t.TYPE_CHECKING:
    from typing_extensions import Unpack

    from sqlglot._typing import GeneratorNoDialectArgs
    from sqlglot.dialects.dialect import DialectType
    from sqlglot.errors import ErrorLevel

    class BoundArgs(GeneratorNoDialectArgs, total=False):
        """Generator options plus the parser's ``error_level``."""

        error_level: ErrorLevel | None


__all__ = ["SOURCE_STYLES", "TARGET_STYLES", "BoundSQL", "transpile_bound"]

SOURCE_STYLES = ("qmark", "named", "numeric")
TARGET_STYLES = ("qmark", "named", "numeric")

# Metadata key used to tag AST nodes that stand for a source parameter marker.
_BIND_META_KEY = "sqlglot_bind_identity"

_NUMERIC_TEXT_RE = re.compile(r"\$([0-9]+)\Z")
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

# A `:name` marker is only a parameter when the colon does not follow something that
# can end an expression or that opens a dialect-specific construct; otherwise the
# colon is most likely an operator (e.g. Snowflake's `col:key`, DuckDB's `arr[:n]`
# slice or prefix aliases).
_EXPRESSION_END_TOKENS = frozenset(
    {
        TokenType.IDENTIFIER,
        TokenType.VAR,
        TokenType.NUMBER,
        TokenType.R_PAREN,
        TokenType.R_BRACKET,
        TokenType.R_BRACE,
        TokenType.STRING,
        TokenType.BIT_STRING,
        TokenType.BYTE_STRING,
        TokenType.HEX_STRING,
        TokenType.RAW_STRING,
        TokenType.NATIONAL_STRING,
        TokenType.NATIONAL_RAW_STRING,
        TokenType.UNICODE_STRING,
        TokenType.HEREDOC_STRING,
        TokenType.TRUE,
        TokenType.FALSE,
        TokenType.NULL,
        TokenType.END,
    }
)

# Bracket balancing help for the lexical named-marker scan.
_OPENING_TOKENS = frozenset({TokenType.L_PAREN, TokenType.L_BRACKET, TokenType.L_BRACE})
_CLOSING_TOKENS = frozenset({TokenType.R_PAREN, TokenType.R_BRACKET, TokenType.R_BRACE})


class BoundSQL:
    """The result of :func:`sqlglot.transpile_bound`.

    Attributes:
        sql: The transpiled SQL statement.
        parameters: The values to bind to ``sql``. For the ``qmark`` and ``numeric``
            target styles this is a tuple ordered by target placeholder, suitable for
            positional binding. For the ``named`` target style it is a new dict keyed
            by ``p1``, ``p2``, ... and ordered by target placeholder.
        parameter_order: The source identity of the parameter at each target slot.
            Identities are 1-based ``int`` values for the ``qmark`` and ``numeric``
            source styles and ``str`` names for the ``named`` source style.

    The attribute order is ``sql``, ``parameters`` and ``parameter_order``; the
    object is not a tuple, but iterating over it yields those three attributes so
    it can be unpacked like the list returned by :func:`sqlglot.transpile`.
    """

    # Keep the public attribute order stable; do not sort this tuple.
    __slots__ = ("sql", "parameters", "parameter_order")  # noqa: RUF023

    def __init__(
        self,
        sql: str,
        parameters: tuple[t.Any, ...] | dict[str, t.Any],
        parameter_order: tuple[t.Any, ...],
    ) -> None:
        self.sql = sql
        self.parameters = parameters
        self.parameter_order = parameter_order

    def __iter__(self) -> t.Iterator[t.Any]:
        yield self.sql
        yield self.parameters
        yield self.parameter_order

    def __repr__(self) -> str:
        # Deliberately omits `parameters`: values must never be echoed in diagnostics.
        return f"BoundSQL(sql={self.sql!r}, parameter_order={self.parameter_order!r})"


def transpile_bound(
    sql: str,
    parameters: t.Any,
    *,
    read: DialectType = None,
    write: DialectType = None,
    source_style: str,
    target_style: str = "qmark",
    **opts: Unpack[BoundArgs],
) -> BoundSQL:
    """Transpiles one SQL statement while tracking its parameters.

    The statement must contain parameter markers written in ``source_style``. The
    generated statement uses markers written in ``target_style``, and the result
    records which source parameter belongs at every target slot.

    Args:
        sql: a single, non-empty SQL statement.
        parameters: the values bound to the source markers. A ``list`` or ``tuple``
            for the ``qmark`` and ``numeric`` source styles, or a ``Mapping`` for the
            ``named`` source style. For ``numeric``, the sequence must cover every
            index up to and including the largest ``$n`` in the statement; unused
            gaps are accepted and are not carried into the target bindings. Input
            sequences and mappings are never modified.
        read: the source dialect used to parse ``sql`` (a dialect name or instance).
        write: the target dialect used to generate SQL (a dialect name or instance).
            Defaults to ``read``.
        source_style: how source parameters are written:

            * ``"qmark"``: ``?`` markers, numbered from 1 by lexical position in
              ``sql``.
            * ``"named"``: ``:name`` markers whose names match
              ``[A-Za-z_][A-Za-z0-9_]*``. Repeating a name refers to one parameter.
            * ``"numeric"``: ``$n`` markers, where ``n`` is a 1-based decimal integer
              without leading zeroes. Repeating an index refers to one parameter.
        target_style: how target parameters are written: ``"qmark"`` (``?``),
            ``"numeric"`` (``$1``, ``$2``, ...) or ``"named"`` (``:p1``, ``:p2``, ...).
            Numbering follows the first use in the generated statement; repeated
            uses of one source parameter reuse the same slot.
        **opts: options forwarded to the generator (``pretty``, ``identify``,
            ``comments``, ``unsupported_level``, ...). ``error_level`` is forwarded to
            the parser, matching :func:`sqlglot.transpile`.

    Returns:
        A :class:`BoundSQL` with ``sql``, ``parameters`` and ``parameter_order``
        attributes. ``parameters`` is a tuple for the ``qmark`` and ``numeric``
        target styles and a dict for ``named``.

    Raises:
        BindError: when the statement is empty or contains multiple statements, the
            style names are unknown, source markers are missing, extra or invalid,
            source styles are mixed, or the parameter container or its contents do
            not match the source markers.
        ParseError: for SQL syntax errors, as with the regular parser.
        UnsupportedError: when the target generator reports unsupported constructs
            at the configured ``unsupported_level``.

    Note:
        This function only transpiles SQL; it does not connect to a database. Choosing
        a target style that the target driver accepts is the caller's responsibility
        (for example, ``sqlite3`` accepts the qmark, named and numeric styles, while
        ``duckdb`` accepts qmark and numeric parameters).

    Example:
        >>> from sqlglot import transpile_bound
        >>> bound = transpile_bound(
        ...     "SELECT STRPOS(?, ?)", ["haystack", "needle"], read="duckdb", write="mysql",
        ...     source_style="qmark", target_style="qmark",
        ... )
        >>> bound.sql
        'SELECT LOCATE(?, ?)'
        >>> bound.parameter_order
        (2, 1)
    """
    if not isinstance(source_style, str) or source_style not in SOURCE_STYLES:
        raise BindError("unknown source_style, expected one of " + ", ".join(SOURCE_STYLES))
    if not isinstance(target_style, str) or target_style not in TARGET_STYLES:
        raise BindError("unknown target_style, expected one of " + ", ".join(TARGET_STYLES))
    if not isinstance(sql, str):
        raise BindError("SQL must be a string")

    read_dialect = Dialect.get_or_raise(read)
    write_dialect = read_dialect if write is None else Dialect.get_or_raise(write)
    error_level = t.cast("ErrorLevel | None", opts.pop("error_level", None))

    tokens = read_dialect.tokenize(sql)
    _reject_empty_or_multiple_statements(tokens)

    named_candidates = _find_named_candidates(tokens)
    numeric_candidates = _find_numeric_candidates(tokens)

    parser, statements, synthetic = _parse_statement(
        tokens, sql, read_dialect, error_level, source_style, named_candidates
    )
    expression = _single_statement(statements)

    # Classify every placeholder the parser actually produced. Lexical positions
    # that the parser consumed as dialect operators (e.g. PostgreSQL's JSON `?`)
    # produce no record. For the named style, only lexically valid `:name` markers
    # count, so constructs such as DuckDB's `arr[:i]` slice stay untouched.
    parser_markers: list[tuple[int, int, exp.Expr, str, str]] = []
    tree_ids = {id(node) for node in expression.walk()}
    named_starts = {start for start, _, _ in named_candidates}
    seen: set[int] = set()
    qmark_records = 0
    invalid_named = False
    for node, consumed in getattr(parser, "_sqlglot_bind_records", ()):
        if not consumed:
            continue
        in_tree = id(node) in tree_ids
        kind, text = _classify_marker(consumed)
        start = min(token.start for token in consumed)
        if kind == "named" and start not in named_starts:
            if in_tree and not _NAME_RE.match(text):
                # e.g. Snowflake's `:1`: a colon parameter whose name is invalid.
                invalid_named = True
            # Otherwise a colon the parser understood as a dialect construct rather
            # than a parameter marker (e.g. a slice or a JSON path accessor).
            continue
        if not in_tree:
            # A placeholder produced while trying an alternative that the parser
            # then abandoned; it is not part of the statement.
            continue
        if kind == "qmark":
            qmark_records += 1
        if id(node) in seen:
            continue
        seen.add(id(node))
        parser_markers.append((start, consumed[-1].end, node, kind, text))

    if invalid_named:
        raise BindError("invalid named parameter marker")

    # Reject markers written in a different style. `?` markers are only recognized
    # through the parser, so dialect operators are not misread as qmark parameters.
    if source_style != "qmark" and qmark_records:
        raise BindError("mixed source parameter styles are not supported")
    if source_style != "numeric" and numeric_candidates:
        raise BindError("mixed source parameter styles are not supported")
    if source_style != "named" and named_candidates:
        raise BindError("mixed source parameter styles are not supported")
    if any(kind == "other" for _, _, _, kind, _ in parser_markers):
        # Dialect-specific parameter syntaxes (widgets, brace parameters, ...) are
        # outside the three supported source styles.
        raise BindError("unsupported source parameter marker")

    if source_style == "numeric":
        candidate_starts = {start for start, _, _ in numeric_candidates}
        if any(
            kind == "numeric" and start not in candidate_starts
            for start, _, _, kind, _ in parser_markers
        ):
            # e.g. `$ 1`: the parser sees a parameter, but it is not written as `$n`.
            raise BindError("invalid numeric parameter marker")

    nodes_by_start = {start: node for start, _, node, _, _ in parser_markers}

    identities: list[tuple[exp.Expr, t.Any]] = []
    qmark_identity = 0
    numeric_seen: set[int] = set()
    named_seen: set[str] = set()

    if source_style == "qmark":
        for _, _, node, kind, _ in sorted(parser_markers, key=lambda item: item[0]):
            if kind != "qmark":
                raise BindError("mixed source parameter styles are not supported")
            qmark_identity += 1
            identities.append((node, qmark_identity))
    elif source_style == "numeric":
        for start, end, digits in numeric_candidates:
            index = _validate_numeric(digits)
            node = nodes_by_start.get(start)
            if node is None:
                node = _identifier_at(expression, (start, end))
            if node is None:
                raise BindError("numeric parameter marker could not be bound")
            numeric_seen.add(index)
            identities.append((node, index))
    else:
        for start, _, name in named_candidates:
            node = nodes_by_start.get(start)
            if node is None and synthetic is not None:
                node = synthetic.get(start)
            if node is None:
                raise BindError("named parameter marker could not be bound")
            if not _NAME_RE.match(name):
                raise BindError("invalid named parameter marker")
            named_seen.add(name)
            identities.append((node, name))

    values = _resolve_values(parameters, source_style, qmark_identity, numeric_seen, named_seen)

    sid_by_identity: dict[t.Any, int] = {}
    for node, identity in identities:
        sid = sid_by_identity.get(identity)
        if sid is None:
            sid = len(sid_by_identity)
            sid_by_identity[identity] = sid
        node.meta[_BIND_META_KEY] = sid
    identity_by_sid = {sid: identity for identity, sid in sid_by_identity.items()}

    nonce = secrets.token_hex(16)
    marker_re = re.compile("\x00" + re.escape(nonce) + r":([0-9]+)\x00")

    generator = write_dialect.generator_class(dialect=write_dialect, **t.cast(t.Any, opts))
    _install_binding_markers(generator, nonce)
    raw_sql = generator.generate(expression, copy=True)

    occurrences = [int(match.group(1)) for match in marker_re.finditer(raw_sql)]
    unknown = [sid for sid in occurrences if sid not in identity_by_sid]
    if unknown:
        raise BindError("generated statement contains an unbound parameter slot")

    target_text: dict[int, str] = {}
    if target_style == "qmark":
        target_parameters: list[t.Any] = []
        parameter_order: list[t.Any] = []
        for sid in occurrences:
            identity = identity_by_sid[sid]
            target_parameters.append(values[identity])
            parameter_order.append(identity)
        result_parameters: tuple[t.Any, ...] | dict[str, t.Any] = tuple(target_parameters)
    else:
        parameter_order = []
        slot_parameters: list[t.Any] = []
        for sid in occurrences:
            if sid in target_text:
                continue
            identity = identity_by_sid[sid]
            slot = len(slot_parameters) + 1
            target_text[sid] = f"${slot}" if target_style == "numeric" else f":p{slot}"
            slot_parameters.append(values[identity])
            parameter_order.append(identity)
        if target_style == "numeric":
            result_parameters = tuple(slot_parameters)
        else:
            result_parameters = {f"p{i + 1}": v for i, v in enumerate(slot_parameters)}

    def replace(match: re.Match[str]) -> str:
        sid = int(match.group(1))
        return "?" if target_style == "qmark" else target_text[sid]

    return BoundSQL(marker_re.sub(replace, raw_sql), result_parameters, tuple(parameter_order))


_parser_class_cache: dict[type, type] = {}


def _parse_statement(
    tokens: list[Token],
    sql: str,
    dialect: Dialect,
    error_level: t.Any,
    source_style: str,
    named_candidates: list[tuple[int, int, str]],
) -> tuple[t.Any, list[exp.Expr | None], dict[int, exp.Expr] | None]:
    parser_class = _bound_parser_class(dialect.parser_class)
    parser = parser_class(
        dialect=dialect, **({"error_level": error_level} if error_level is not None else {})
    )

    # A fused `?::` token would hide a qmark marker, so it is split before parsing.
    # This applies regardless of the declared source style so that a qmark marker
    # used with the wrong style is still detected as a mixed style.
    parse_tokens = _split_qdcolon(tokens)

    try:
        statements = parser.parse(parse_tokens, sql)
    except ParseError:
        if source_style != "named":
            raise
        statements = None

    if source_style == "named":
        records = getattr(parser, "_sqlglot_bind_records", ())
        bound_starts = {
            min(token.start for token in consumed) for _, consumed in records if consumed
        }
        if statements is None or any(start not in bound_starts for start, _, _ in named_candidates):
            # Some dialects cannot parse `:name` everywhere (for example DuckDB reads
            # `:x` as a prefix alias at the start of an expression). Retry with the
            # markers rewritten to private identifiers, then map those identifiers
            # back to the named parameters.
            rewritten, span_by_name = _rewrite_named_markers(tokens, named_candidates)
            retry = parser_class(
                dialect=dialect,
                **({"error_level": error_level} if error_level is not None else {}),
            )
            try:
                statements = retry.parse(rewritten, sql)
            except ParseError:
                if statements is None:
                    raise
                return parser, statements, None
            parser = retry

            synthetic: dict[int, exp.Expr] = {}
            tree_ids = {
                id(node) for node in (statements[0].walk() if statements and statements[0] else ())
            }
            for node in statements[0].walk() if statements and statements[0] else ():
                if not isinstance(node, exp.Identifier):
                    continue
                span = span_by_name.get(node.name)
                if span is not None and id(node) in tree_ids:
                    synthetic[span] = node

            return parser, statements, synthetic

    return parser, statements or [], None


def _single_statement(statements: list[exp.Expr | None]) -> exp.Expr:
    expressions = [statement for statement in statements if statement is not None]
    if not expressions:
        raise BindError("expected one non-empty SQL statement, got none")
    if len(expressions) > 1:
        raise BindError("expected a single SQL statement, got multiple")
    return expressions[0]


def _bound_parser_class(base: type) -> type:
    """Returns a parser subclass that records the placeholder it just parsed."""
    cached = _parser_class_cache.get(base)
    if cached is not None:
        return cached

    class BoundParser(base):  # type: ignore[misc, valid-type]
        __slots__ = ("_sqlglot_bind_records",)

        def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
            super().__init__(*args, **kwargs)
            self._sqlglot_bind_records: list[tuple[exp.Expr, list[Token]]] = []

        def _parse_placeholder(self) -> t.Any:
            start = self._index
            node = super()._parse_placeholder()
            if node is not None:
                self._sqlglot_bind_records.append((node, self._tokens[start : self._index]))
            return node

    BoundParser.__name__ = f"Bound{base.__name__}"
    BoundParser.__qualname__ = BoundParser.__name__
    _parser_class_cache[base] = BoundParser
    return BoundParser


def _install_binding_markers(generator: t.Any, nonce: str) -> None:
    """Teaches a generator instance to emit resolvable markers for tagged nodes.

    The generator dispatches ``_sql`` handlers through its ``_dispatch`` dict, so the
    instance's dispatch table is copied and wrapped. Child generators (as used by the
    Athena generator, whose ``generate`` delegates) are patched too.
    """
    prefix = f"\x00{nonce}:"

    def wrap(handler: t.Callable[[t.Any, exp.Expr], str]) -> t.Callable[[t.Any, exp.Expr], str]:
        def marker_handler(generator: t.Any, expression: exp.Expr) -> str:
            sid = expression.meta_get(_BIND_META_KEY)
            if sid is not None:
                return f"{prefix}{sid}\x00"
            return handler(generator, expression)

        return marker_handler

    for candidate in _iter_generators(generator):
        dispatch = dict(candidate._dispatch)
        for node_class in (exp.Placeholder, exp.Parameter, exp.Identifier):
            handler = dispatch.get(node_class)
            if handler is not None:
                dispatch[node_class] = wrap(handler)
        candidate._dispatch = dispatch


def _iter_generators(generator: t.Any) -> t.Iterator[t.Any]:
    """Yields the generator and any generator instances stored in its slots."""
    from sqlglot.generator import Generator

    visited: set[int] = set()
    stack = [generator]
    while stack:
        current = stack.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        yield current

        for cls in type(current).__mro__:
            for slot in getattr(cls, "__slots__", ()):
                if not isinstance(slot, str):
                    continue
                value = getattr(current, slot, None)
                if isinstance(value, Generator) and id(value) not in visited:
                    stack.append(value)


def _split_qdcolon(tokens: list[Token]) -> list[Token]:
    """Splits the fused ``?::`` token so the ``?`` can be a qmark parameter."""
    if not any(token.token_type == TokenType.QDCOLON for token in tokens):
        return tokens

    result: list[Token] = []
    for token in tokens:
        if token.token_type != TokenType.QDCOLON:
            result.append(token)
            continue

        result.append(
            Token(
                TokenType.PLACEHOLDER,
                "?",
                token.line,
                token.col,
                token.start,
                token.start,
                list(token.comments),
            )
        )
        result.append(
            Token(TokenType.DCOLON, "::", token.line, token.col + 1, token.start + 1, token.end, [])
        )
    return result


def _reject_empty_or_multiple_statements(tokens: list[Token]) -> None:
    semicolons = [
        index for index, token in enumerate(tokens) if token.token_type == TokenType.SEMICOLON
    ]
    if not semicolons:
        if not tokens:
            raise BindError("expected one non-empty SQL statement, got none")
        return

    if len(semicolons) > 1:
        raise BindError("expected a single SQL statement, got multiple")

    semicolon = semicolons[0]
    if any(token.token_type != TokenType.SEMICOLON for token in tokens[semicolon + 1 :]):
        raise BindError("expected a single SQL statement, got multiple")
    if not any(token.token_type != TokenType.SEMICOLON for token in tokens[:semicolon]):
        raise BindError("expected one non-empty SQL statement, got none")


def _classify_marker(consumed: list[Token]) -> tuple[str, str]:
    """Classifies the tokens a placeholder parser consumed."""
    first = consumed[0]

    if first.token_type == TokenType.PLACEHOLDER:
        return ("qmark", "")

    if first.token_type == TokenType.PARAMETER:
        if len(consumed) == 2 and consumed[1].token_type == TokenType.NUMBER:
            return ("numeric", consumed[1].text)
        return ("other", "")

    if first.token_type == TokenType.COLON:
        return ("named", "".join(token.text for token in consumed[1:]))

    # Dialect-specific placeholder syntaxes (widgets, brace parameters, ...) are
    # not part of the three supported source styles.
    return ("other", "")


def _find_named_candidates(tokens: list[Token]) -> list[tuple[int, int, str]]:
    """Finds lexical ``:name`` markers that are not dialect path/alias operators."""
    candidates: list[tuple[int, int, str]] = []
    previous: Token | None = None

    depth = 0
    for index, token in enumerate(tokens):
        token_type = token.token_type
        if token_type in _OPENING_TOKENS:
            depth += 1
        elif token_type in _CLOSING_TOKENS:
            depth = max(depth - 1, 0)

        if token_type != TokenType.COLON:
            previous = token
            continue

        if index + 1 >= len(tokens):
            previous = token
            continue

        name_token = tokens[index + 1]
        name = name_token.text
        # Names may coincide with keywords, so any token whose text looks like an
        # identifier is accepted here.
        if not _NAME_RE.match(name):
            previous = token
            continue

        if previous is not None and previous.token_type in _EXPRESSION_END_TOKENS:
            previous = token
            continue

        # Inside brackets (e.g. DuckDB's `arr[:i]` slice) a leading colon is an
        # operator, not a parameter marker.
        if depth and previous is not None and previous.token_type == TokenType.L_BRACKET:
            previous = token
            continue

        candidates.append((token.start, name_token.end, name))
        previous = name_token

    return candidates


def _find_numeric_candidates(tokens: list[Token]) -> list[tuple[int, int, str]]:
    """Finds lexical ``$n`` markers, including ones a dialect reads as columns."""
    candidates: list[tuple[int, int, str]] = []

    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token.token_type == TokenType.VAR and token.text.startswith("$"):
            match = _NUMERIC_TEXT_RE.match(token.text)
            if match:
                candidates.append((token.start, token.end, match.group(1)))
            index += 1
            continue

        if (
            token.token_type == TokenType.PARAMETER
            and token.text == "$"
            and index + 1 < len(tokens)
            and tokens[index + 1].token_type == TokenType.NUMBER
            and tokens[index + 1].start == token.end + 1
        ):
            number = tokens[index + 1]
            match = _NUMERIC_TEXT_RE.match("$" + number.text)
            if not match:
                raise BindError("invalid numeric parameter marker")
            candidates.append((token.start, number.end, match.group(1)))
            index += 2
            continue

        index += 1

    return candidates


def _validate_numeric(digits: str) -> int:
    if not digits or not digits.isascii() or not digits.isdigit():
        raise BindError("invalid numeric parameter marker")
    if digits[0] == "0":
        raise BindError("invalid numeric parameter marker: indices have no leading zeroes")
    return int(digits)


def _identifier_at(expression: exp.Expr, span: tuple[int, int]) -> exp.Expr | None:
    """Returns the identifier node that was parsed from a lexical token span."""
    for node in expression.walk():
        if (
            isinstance(node, exp.Identifier)
            and (
                node.meta_get("start"),
                node.meta_get("end"),
            )
            == span
        ):
            return node
    return None


def _rewrite_named_markers(
    tokens: list[Token], named_candidates: list[tuple[int, int, str]]
) -> tuple[list[Token], dict[str, int]]:
    """Rewrites ``:name`` markers to private identifiers, indexed by colon position."""
    spans = {start: name for start, _, name in named_candidates}
    span_by_name: dict[str, int] = {}
    result: list[Token] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]
        if token.start in spans and index + 1 < len(tokens):
            synthetic = f"__sqlglot_bind_{len(span_by_name)}__"
            span_by_name[synthetic] = token.start
            result.append(
                Token(
                    TokenType.IDENTIFIER,
                    synthetic,
                    token.line,
                    token.col,
                    token.start,
                    tokens[index + 1].end,
                )
            )
            index += 2
            continue

        result.append(token)
        index += 1

    return result, span_by_name


def _resolve_values(
    parameters: t.Any,
    source_style: str,
    qmark_count: int,
    numeric_seen: set[int],
    named_seen: set[str],
) -> dict[t.Any, t.Any]:
    """Validates the parameter container and maps source identities to values."""
    if source_style in ("qmark", "numeric"):
        if not isinstance(parameters, (list, tuple)):
            raise BindError(
                f"parameters must be a list or tuple for the {source_style} source style"
            )

        expected = (
            qmark_count if source_style == "qmark" else (max(numeric_seen) if numeric_seen else 0)
        )
        if len(parameters) != expected:
            raise BindError(
                f"wrong number of parameters: expected {expected}, got {len(parameters)}"
            )

        values: dict[t.Any, t.Any] = {}
        if source_style == "qmark":
            for index in range(1, qmark_count + 1):
                values[index] = parameters[index - 1]
        else:
            # Gaps are deliberately skipped: their input values are unused.
            for index in sorted(numeric_seen):
                values[index] = parameters[index - 1]
        return values

    if not isinstance(parameters, Mapping):
        raise BindError("parameters must be a mapping for the named source style")

    keys = set(parameters)
    if keys != named_seen:
        # Diagnostics intentionally mention counts only: names and values never
        # appear in the message.
        raise BindError(
            "parameter names do not match the statement "
            f"({len(named_seen - keys)} missing, {len(keys - named_seen)} extra)"
        )

    return {name: parameters[name] for name in named_seen}
