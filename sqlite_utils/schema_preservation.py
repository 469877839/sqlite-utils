"""Strict preservation analysis for :meth:`Table.transform`.

``transform()`` has to rebuild a table, and SQLite drops every index and
trigger attached to that table while dependent views keep pointing at the
old column names as stored text.  The lenient behaviour is to preserve what
is easy and raise for the rest; this module powers an opt-in strict mode
that analyses every dependent object *before* any destructive SQL runs.

Every rewrite here works on tokens produced by
:mod:`sqlite_utils.create_table_parser` - strings, quoted identifiers and
comments are opaque, so definitions like ``'name'`` literals or quoted
column names are never mistaken for references.  Definitions that cannot be
migrated reliably raise :class:`PreservationError` instead of being silently
dropped or mangled.
"""

from dataclasses import dataclass, field
from typing import Any

from .create_table_parser import (
    ParseError,
    _ascii_fold,
    _is_identifier_token,
    _lex,
    _meaningful,
    _matching_paren,
    _quote_replacement,
    _split_spans,
    _unquote,
    _valid_bare_identifier,
)


class PreservationError(Exception):
    """Raised when strict preservation cannot guarantee a faithful migration."""


@dataclass
class _Analysis:
    # SQL fragments to apply to the original definition.
    edits: list[tuple[int, int, str]] = field(default_factory=list)

    def rewritten(self, sql: str) -> str:
        for start, end, replacement in sorted(
            self.edits, key=lambda item: item[0], reverse=True
        ):
            sql = sql[:start] + replacement + sql[end:]
        return sql


def _folded(names: set[str]) -> set[str]:
    return {_ascii_fold(name) for name in names}


def _renamed_map(rename: dict[str, str]) -> dict[str, str]:
    return {_ascii_fold(old): new for old, new in rename.items()}


def _refuse(object_type: str, name: str, reason: str) -> None:
    raise PreservationError(
        f"Cannot transform table while strict preservation is enabled: "
        f"{object_type} {name!r} {reason}. Resolve this manually (drop or "
        "recreate the object, or transform without strict preservation)."
    )


def _is_column_token(tokens: list[Any], index: int) -> bool:
    token = tokens[index]
    if not _is_identifier_token(tokens, index):
        return False
    if token.kind == "identifier":
        return True
    return _valid_bare_identifier(token.text)


def _edit(analysis: _Analysis, token: Any, new_name: str) -> None:
    analysis.edits.append(
        (token.start, token.end, _quote_replacement(token.text, new_name))
    )


@dataclass(frozen=True)
class _Reference:
    qualifier: str  # "" for a bare column reference, else NEW/OLD/table name
    name: str
    token: Any  # token holding ``name``


def _scan_expression(expression: str) -> list[_Reference]:
    """Token-level scan of an expression for column references.

    Qualified references (``NEW.a``, ``other.a``) are reported with their
    qualifier; the qualifier token itself is not reported.  Strings, quoted
    identifiers used as strings, function names and SQL keywords are never
    reported.
    """
    tokens = _meaningful(_lex(expression))
    references: list[_Reference] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            _is_identifierish(token)
            and index + 2 < len(tokens)
            and tokens[index + 1].text == "."
            and _is_identifierish(tokens[index + 2])
        ):
            qualifier = _unquote(token.text)
            name_token = tokens[index + 2]
            references.append(
                _Reference(qualifier, _unquote(name_token.text), name_token)
            )
            index += 3
            continue
        if _is_column_token(tokens, index):
            references.append(_Reference("", _unquote(token.text), token))
        index += 1
    return references


def _is_identifierish(token: Any) -> bool:
    if token.kind == "identifier":
        return True
    if token.kind != "word":
        return False
    return _valid_bare_identifier(token.text)


def _qualified_name(tokens: list[Any], index: int) -> tuple[str, int]:
    """Read ``name`` or ``schema.name`` and return the name plus next index."""
    if index >= len(tokens) or not _is_identifierish(tokens[index]):
        raise ParseError("Expected an object name")
    name = _unquote(tokens[index].text)
    if (
        index + 2 < len(tokens)
        and tokens[index + 1].text == "."
        and _is_identifierish(tokens[index + 2])
    ):
        return _unquote(tokens[index + 2].text), index + 3
    return name, index + 1


def _check_column_reference(
    object_type: str,
    object_name: str,
    reference: _Reference,
    columns_folded: set[str],
    drop_folded: set[str],
    rename_folded: dict[str, str],
    analysis: _Analysis,
    allow_qualified: bool = False,
) -> None:
    """Validate one reference, recording an edit when it must be renamed."""
    if reference.qualifier:
        if not allow_qualified:
            _refuse(
                object_type,
                object_name,
                f"references {reference.qualifier}.{reference.name} in a context "
                "where only columns of the transformed table may be used",
            )
        return
    folded = _ascii_fold(reference.name)
    if folded in drop_folded:
        _refuse(
            object_type,
            object_name,
            f"depends on column {reference.name!r} which is being dropped",
        )
    if folded in rename_folded:
        _edit(analysis, reference.token, rename_folded[folded])
    elif folded not in columns_folded:
        _refuse(
            object_type,
            object_name,
            f"references unknown column {reference.name!r}",
        )


def _simple_index_column(item: str) -> Any | None:
    """Return the column token if ``item`` is a plain ``col [COLLATE x] [DESC]``."""
    tokens = _meaningful(_lex(item))
    if not tokens or not _is_identifierish(tokens[0]):
        return None
    index = 1
    if index < len(tokens) and tokens[index].is_keyword("COLLATE"):
        if index + 1 >= len(tokens) or not _is_identifierish(tokens[index + 1]):
            return None
        index += 2
    if index < len(tokens) and (
        tokens[index].is_keyword("ASC") or tokens[index].is_keyword("DESC")
    ):
        index += 1
    if index != len(tokens):
        return None
    return tokens[0]


def _parse_index_sql(sql: str, table_name: str) -> tuple[str, str, int, int]:
    """Validate a CREATE INDEX statement.

    Returns ``(index_name, indexed_body, body_start, where_start)`` where
    ``indexed_body`` is the text between the parentheses, ``body_start`` its
    offset in ``sql`` and ``where_start`` the offset of the WHERE keyword or
    ``-1`` when the index is not partial.
    """
    tokens = _meaningful(_lex(sql))
    index = 0
    if index >= len(tokens) or not tokens[index].is_keyword("CREATE"):
        raise ParseError("Expected CREATE INDEX")
    index += 1
    if index < len(tokens) and tokens[index].is_keyword("UNIQUE"):
        index += 1
    if index >= len(tokens) or not tokens[index].is_keyword("INDEX"):
        raise ParseError("Expected CREATE INDEX")
    index += 1
    if (
        index + 2 < len(tokens)
        and tokens[index].is_keyword("IF")
        and tokens[index + 1].is_keyword("NOT")
        and tokens[index + 2].is_keyword("EXISTS")
    ):
        index += 3
    index_name, index = _qualified_name(tokens, index)
    if index >= len(tokens) or not tokens[index].is_keyword("ON"):
        raise ParseError("Expected ON after the index name")
    index += 1
    index_table, index = _qualified_name(tokens, index)
    if _ascii_fold(index_table) != _ascii_fold(table_name):
        raise ParseError(
            f"Index is defined on table {index_table!r}, not {table_name!r}"
        )
    if index >= len(tokens) or tokens[index].text != "(":
        raise ParseError("Expected an indexed column list")
    close = _matching_paren(tokens, index)
    body_start = tokens[index].end
    body_end = tokens[close].start
    where_start = -1
    trailing = tokens[close + 1 :]
    if trailing:
        if not trailing[0].is_keyword("WHERE"):
            raise ParseError("Unexpected SQL after the indexed column list")
        where_start = trailing[0].start
    return index_name, sql[body_start:body_end], body_start, where_start


def _analyze_namespace(
    object_type: str,
    object_name: str,
    expression: str,
    columns_folded: set[str],
    drop_folded: set[str],
    rename_folded: dict[str, str],
    analysis: _Analysis,
) -> None:
    for reference in _scan_expression(expression):
        _check_column_reference(
            object_type,
            object_name,
            reference,
            columns_folded,
            drop_folded,
            rename_folded,
            analysis,
        )


def plan_index(
    index_name: str,
    sql: str,
    table_name: str,
    columns: set[str],
    rename: dict[str, str],
    drop: set[str],
) -> str:
    """Return CREATE INDEX SQL for the rebuilt table, or raise PreservationError."""
    try:
        _, body, body_start, where_start = _parse_index_sql(sql, table_name)
    except ParseError as ex:
        _refuse("index", index_name, f"does not have a parseable CREATE INDEX ({ex})")
    columns_folded = _folded(columns)
    drop_folded = _folded(drop)
    rename_folded = _renamed_map(rename)
    edits: list[tuple[int, int, str]] = []
    for item, item_start, _ in _split_spans(body, _lex(body)):
        analysis = _Analysis()
        column_token = _simple_index_column(item)
        if column_token is not None:
            reference = _Reference("", _unquote(column_token.text), column_token)
            _check_column_reference(
                "index",
                index_name,
                reference,
                columns_folded,
                drop_folded,
                rename_folded,
                analysis,
            )
        else:
            _analyze_namespace(
                "index",
                index_name,
                item,
                columns_folded,
                drop_folded,
                rename_folded,
                analysis,
            )
        edits.extend(
            (body_start + item_start + start, body_start + item_start + end, text)
            for start, end, text in analysis.edits
        )
    if where_start != -1:
        where_text = sql[where_start + len("WHERE") :]
        analysis = _Analysis()
        _analyze_namespace(
            "index",
            index_name,
            where_text,
            columns_folded,
            drop_folded,
            rename_folded,
            analysis,
        )
        where_offset = where_start + len("WHERE")
        edits.extend(
            (where_offset + start, where_offset + end, text)
            for start, end, text in analysis.edits
        )
    return _Analysis(edits=edits).rewritten(sql)


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------

_TRIGGER_EVENTS = frozenset(("DELETE", "INSERT", "UPDATE"))


@dataclass(frozen=True)
class _TriggerHeader:
    name: str
    event: str
    event_table: str
    # Indices into the meaningful-token list.
    of_first: int  # first UPDATE OF column token, or -1
    of_last: int  # last UPDATE OF column token, or -1
    when_first: int  # first WHEN expression token, or -1
    when_last: int  # last WHEN expression token (exclusive), or -1
    begin_index: int  # BEGIN keyword token, or -1 for a single-statement body
    first_body_index: int  # first body statement token


def _parse_trigger_header(sql: str) -> _TriggerHeader:
    tokens = _meaningful(_lex(sql))
    index = 0
    if index >= len(tokens) or not tokens[index].is_keyword("CREATE"):
        raise ParseError("Expected CREATE TRIGGER")
    index += 1
    if index < len(tokens) and (
        tokens[index].is_keyword("TEMP") or tokens[index].is_keyword("TEMPORARY")
    ):
        index += 1
    if index >= len(tokens) or not tokens[index].is_keyword("TRIGGER"):
        raise ParseError("Expected CREATE TRIGGER")
    index += 1
    if (
        index + 2 < len(tokens)
        and tokens[index].is_keyword("IF")
        and tokens[index + 1].is_keyword("NOT")
        and tokens[index + 2].is_keyword("EXISTS")
    ):
        index += 3
    name, index = _qualified_name(tokens, index)
    if index < len(tokens) and tokens[index].is_keyword("INSTEAD"):
        if index + 1 >= len(tokens) or not tokens[index + 1].is_keyword("OF"):
            raise ParseError("INSTEAD must be followed by OF")
        index += 2
    elif index < len(tokens) and (
        tokens[index].is_keyword("BEFORE") or tokens[index].is_keyword("AFTER")
    ):
        index += 1
    if index >= len(tokens) or tokens[index].text.upper() not in _TRIGGER_EVENTS:
        raise ParseError("Expected a trigger event (DELETE, INSERT or UPDATE)")
    event = tokens[index].text.upper()
    index += 1
    of_first = of_last = -1
    if event == "UPDATE" and index < len(tokens) and tokens[index].is_keyword("OF"):
        index += 1
        of_first = index
        while index < len(tokens) and not tokens[index].is_keyword("ON"):
            index += 1
        if index >= len(tokens):
            raise ParseError("UPDATE OF list is not followed by ON")
        of_last = index
    if index >= len(tokens) or not tokens[index].is_keyword("ON"):
        raise ParseError("Expected ON after the trigger event")
    index += 1
    event_table, index = _qualified_name(tokens, index)
    if (
        index < len(tokens)
        and tokens[index].is_keyword("FOR")
        and index + 1 < len(tokens)
        and tokens[index + 1].is_keyword("EACH")
        and index + 2 < len(tokens)
        and tokens[index + 2].is_keyword("ROW")
    ):
        index += 3
    when_first = when_last = -1
    if index < len(tokens) and tokens[index].is_keyword("WHEN"):
        index += 1
        when_first = index
        while index < len(tokens) and not tokens[index].is_keyword("BEGIN"):
            index += 1
        if index >= len(tokens):
            raise ParseError("WHEN clause is not followed by BEGIN")
        when_last = index
    begin_index = -1
    if index < len(tokens) and tokens[index].is_keyword("BEGIN"):
        begin_index = index
        index += 1
    if index >= len(tokens):
        raise ParseError("Trigger body is empty")
    return _TriggerHeader(
        name=name,
        event=event,
        event_table=event_table,
        of_first=of_first,
        of_last=of_last,
        when_first=when_first,
        when_last=when_last,
        begin_index=begin_index,
        first_body_index=index,
    )


def _split_statements(tokens: list[Any]) -> list[list[Any]]:
    """Split a trigger body token list on top-level semicolons."""
    statements: list[list[Any]] = []
    current: list[Any] = []
    depth = 0
    for token in tokens:
        if token.text == "(":
            depth += 1
        elif token.text == ")":
            depth -= 1
        if token.text == ";" and depth == 0:
            if current:
                statements.append(current)
                current = []
            continue
        current.append(token)
    if depth:
        raise ParseError("Unbalanced parentheses in trigger body")
    if current:
        statements.append(current)
    if not statements:
        raise ParseError("Trigger body is empty")
    return statements


_PSEUDO_COLUMNS = frozenset(("rowid", "oid", "_rowid_"))


def _analyze_trigger_reference(
    object_type: str,
    object_name: str,
    reference: _Reference,
    table_name_folded: str,
    columns_folded: set[str],
    drop_folded: set[str],
    rename_folded: dict[str, str],
    analysis: _Analysis,
    table_scope: bool,
    in_subquery: bool,
    own_table: bool = True,
    event_view_depends_on_table: bool = False,
) -> None:
    folded = _ascii_fold(reference.name)
    if folded in _PSEUDO_COLUMNS:
        return
    if reference.qualifier:
        qualifier = _ascii_fold(reference.qualifier)
        if qualifier in ("new", "old"):
            if not own_table:
                if event_view_depends_on_table and (
                    folded in drop_folded or folded in rename_folded
                ):
                    _refuse(
                        object_type,
                        object_name,
                        f"references {reference.qualifier}.{reference.name} on a "
                        "view over the transformed table; the view's column "
                        "names change with the table and the trigger cannot be "
                        "rewritten reliably",
                    )
                # NEW./OLD. otherwise belong to the trigger's own table or
                # view, not the transformed one.
                return
            if folded in drop_folded:
                _refuse(
                    object_type,
                    object_name,
                    f"references {reference.qualifier}.{reference.name} "
                    "but that column is being dropped",
                )
            if folded in rename_folded:
                _edit(analysis, reference.token, rename_folded[folded])
            elif folded not in columns_folded:
                _refuse(
                    object_type,
                    object_name,
                    "references unknown column "
                    f"{reference.qualifier}.{reference.name}",
                )
            return
        if qualifier == table_name_folded:
            table_scope = True
        else:
            # A different table or alias owns this reference.
            return
    if in_subquery:
        if folded in drop_folded or folded in rename_folded:
            _refuse(
                object_type,
                object_name,
                f"references column {reference.name!r} inside a subquery whose "
                "table scope cannot be determined reliably",
            )
        return
    if not table_scope:
        if folded in drop_folded or folded in rename_folded:
            _refuse(
                object_type,
                object_name,
                f"references column {reference.name!r} in a statement that does "
                "not target the transformed table; its ownership cannot be "
                "determined reliably",
            )
        return
    if folded in drop_folded:
        _refuse(
            object_type,
            object_name,
            f"depends on column {reference.name!r} which is being dropped",
        )
    if folded in rename_folded:
        _edit(analysis, reference.token, rename_folded[folded])
    elif folded not in columns_folded:
        _refuse(
            object_type,
            object_name,
            f"references unknown column {reference.name!r}",
        )


def _reference_ranges(
    tokens: list[Any], start: int, end: int
) -> list[tuple[_Reference, bool, int]]:
    """Return references in tokens[start:end] with an in-subquery flag
    and the index of the reference's name token.

    A parenthesised region counts as a subquery when it begins with SELECT.
    Qualifier tokens and the dot between ``table.column`` are skipped.
    """
    results: list[tuple[_Reference, bool, int]] = []
    depth = 0
    subquery_stack: list[int] = []
    index = start
    while index < end:
        token = tokens[index]
        if token.text == "(":
            depth += 1
            if (
                index + 1 < end
                and tokens[index + 1].is_keyword("SELECT")
            ):
                subquery_stack.append(depth)
            index += 1
            continue
        if token.text == ")":
            if subquery_stack and depth == subquery_stack[-1]:
                subquery_stack.pop()
            depth -= 1
            index += 1
            continue
        in_subquery = bool(subquery_stack)
        if (
            _is_identifierish(token)
            and index + 2 < end
            and tokens[index + 1].text == "."
            and _is_identifierish(tokens[index + 2])
        ):
            results.append(
                (
                    _Reference(
                        _unquote(token.text),
                        _unquote(tokens[index + 2].text),
                        tokens[index + 2],
                    ),
                    in_subquery,
                    index + 2,
                )
            )
            index += 3
            continue
        if _is_column_token(tokens, index):
            results.append(
                (_Reference("", _unquote(token.text), token), in_subquery, index)
            )
        index += 1
    if depth:
        raise ParseError("Unbalanced parentheses in trigger statement")
    return results


def _target_table(tokens: list[Any], keyword_index: int) -> tuple[str, int, int]:
    """Return the target table name and the token range it occupies."""
    index = keyword_index + 1
    # INSERT OR REPLACE INTO ...
    if tokens[keyword_index].is_keyword("INSERT") and index < len(tokens):
        if tokens[index].is_keyword("OR"):
            index += 2
        if index < len(tokens) and tokens[index].is_keyword("INTO"):
            index += 1
    elif tokens[keyword_index].is_keyword("DELETE"):
        if index < len(tokens) and tokens[index].is_keyword("FROM"):
            index += 1
    elif tokens[keyword_index].is_keyword("UPDATE"):
        if index < len(tokens) and tokens[index].is_keyword("OR"):
            index += 2
    name, next_index = _qualified_name(tokens, index)
    return name, index, next_index


def _analyze_statement(
    object_type: str,
    object_name: str,
    tokens: list[Any],
    table_name_folded: str,
    columns_folded: set[str],
    drop_folded: set[str],
    rename_folded: dict[str, str],
    analysis: _Analysis,
    own_table: bool,
    event_view_depends_on_table: bool,
) -> None:
    head = tokens[0].text.upper() if tokens else ""
    target_folded: str | None = None
    skip_start = skip_end = -1
    if head in ("UPDATE", "INSERT", "DELETE"):
        target, skip_start, skip_end = _target_table(tokens, 0)
        target_folded = _ascii_fold(target)
    table_scope = target_folded == table_name_folded
    for reference, in_subquery, token_index in _reference_ranges(
        tokens, 0, len(tokens)
    ):
        if skip_start != -1 and skip_start <= token_index < skip_end:
            continue
        _analyze_trigger_reference(
            object_type,
            object_name,
            reference,
            table_name_folded,
            columns_folded,
            drop_folded,
            rename_folded,
            analysis,
            table_scope=table_scope,
            in_subquery=in_subquery,
            own_table=own_table,
            event_view_depends_on_table=event_view_depends_on_table,
        )


def plan_trigger(
    trigger_name: str,
    sql: str,
    table_name: str,
    columns: set[str],
    rename: dict[str, str],
    drop: set[str],
    dependent_views: set[str] | None = None,
) -> tuple[bool, str]:
    """Return ``(touched, new_sql)`` for a trigger, or raise PreservationError.

    Triggers defined on the transformed table and triggers defined on other
    tables that write to it are both handled.  ``dependent_views`` is the
    set of view names (folded) that reference the transformed table; an
    INSTEAD OF trigger on such a view cannot have its NEW./OLD. references
    rewritten reliably and is refused instead.  ``touched`` is False when
    the definition contains no reference to a renamed or dropped column.
    """
    try:
        header = _parse_trigger_header(sql)
    except ParseError as ex:
        _refuse("trigger", trigger_name, f"does not have a parseable definition ({ex})")
    columns_folded = _folded(columns)
    drop_folded = _folded(drop)
    rename_folded = _renamed_map(rename)
    table_name_folded = _ascii_fold(table_name)
    own_table = _ascii_fold(header.event_table) == table_name_folded
    event_view_depends_on_table = (
        not own_table
        and dependent_views is not None
        and _ascii_fold(header.event_table) in dependent_views
    )
    tokens = _meaningful(_lex(sql))
    analysis = _Analysis()

    def check_ref(reference: _Reference, table_scope: bool, in_subquery: bool) -> None:
        _analyze_trigger_reference(
            "trigger",
            trigger_name,
            reference,
            table_name_folded,
            columns_folded,
            drop_folded,
            rename_folded,
            analysis,
            table_scope=table_scope,
            in_subquery=in_subquery,
            own_table=own_table,
            event_view_depends_on_table=event_view_depends_on_table,
        )

    if own_table:
        # UPDATE OF col1, col2 - every name is a column of this table.
        if header.of_first != -1:
            for index in range(header.of_first, header.of_last):
                if _is_identifierish(tokens[index]):
                    check_ref(
                        _Reference(
                            "", _unquote(tokens[index].text), tokens[index]
                        ),
                        table_scope=True,
                        in_subquery=False,
                    )
        # WHEN clause - only NEW/OLD references can be bound.
        if header.when_first != -1:
            when_tokens = tokens[header.when_first : header.when_last]
            for reference, in_subquery, _ in _reference_ranges(
                when_tokens, 0, len(when_tokens)
            ):
                if not reference.qualifier:
                    folded = _ascii_fold(reference.name)
                    if folded in drop_folded or folded in rename_folded:
                        _refuse(
                            "trigger",
                            trigger_name,
                            "uses unqualified column "
                            f"{reference.name!r} in its WHEN clause; SQLite only "
                            "binds NEW./OLD. references there",
                        )
                    continue
                check_ref(
                    reference, table_scope=False, in_subquery=in_subquery
                )
    # Body statements.
    body_end = len(tokens)
    if header.begin_index != -1:
        last = len(tokens) - 1
        if last >= 0 and tokens[last].text == ";":
            last -= 1
        if last < 0 or not tokens[last].is_keyword("END"):
            _refuse(
                "trigger",
                trigger_name,
                "does not end with END",
            )
        body_end = last
    body_tokens = tokens[header.first_body_index : body_end]
    try:
        statements = _split_statements(body_tokens)
    except ParseError as ex:
        _refuse("trigger", trigger_name, str(ex))
    for statement in statements:
        _analyze_statement(
            "trigger",
            trigger_name,
            statement,
            table_name_folded,
            columns_folded,
            drop_folded,
            rename_folded,
            analysis,
            own_table=own_table,
            event_view_depends_on_table=event_view_depends_on_table,
        )
    return bool(analysis.edits), analysis.rewritten(sql)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

# Clauses that end a FROM / JOIN table specification at parenthesis depth 0.
_FROM_CLAUSE_KEYWORDS = frozenset(
    (
        "WHERE",
        "GROUP",
        "HAVING",
        "WINDOW",
        "ORDER",
        "LIMIT",
        "OFFSET",
        "UNION",
        "EXCEPT",
        "INTERSECT",
    )
)


def _parse_view_header(sql: str) -> tuple[str, int]:
    """Return the view name and the token index after its ``AS`` keyword."""
    tokens = _meaningful(_lex(sql))
    index = 0
    if index >= len(tokens) or not tokens[index].is_keyword("CREATE"):
        raise ParseError("Expected CREATE VIEW")
    index += 1
    if index < len(tokens) and (
        tokens[index].is_keyword("TEMP") or tokens[index].is_keyword("TEMPORARY")
    ):
        index += 1
    if index >= len(tokens) or not tokens[index].is_keyword("VIEW"):
        raise ParseError("Expected CREATE VIEW")
    index += 1
    if (
        index + 2 < len(tokens)
        and tokens[index].is_keyword("IF")
        and tokens[index + 1].is_keyword("NOT")
        and tokens[index + 2].is_keyword("EXISTS")
    ):
        index += 3
    name, index = _qualified_name(tokens, index)
    if index >= len(tokens) or not tokens[index].is_keyword("AS"):
        raise ParseError("Expected AS after the view name")
    return name, index + 1


def _collect_from_sources(
    tokens: list[Any], table_name_folded: str
) -> tuple[set[str], bool, set[int]]:
    """Scan FROM/JOIN sources for the transformed table.

    Returns ``(aliases, single_source, source_indices)`` where ``aliases``
    contains every bare name or alias that unambiguously refers to the
    transformed table, ``single_source`` is True when no second table source
    is visible and ``source_indices`` holds token positions occupied by
    source names and aliases (those tokens are table names, not columns).
    """
    aliases: set[str] = set()
    source_indices: set[int] = set()
    source_count = 0
    in_from = False
    need_source = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        upper = token.text.upper()
        if token.is_keyword("FROM"):
            in_from = True
            need_source = True
            index += 1
            continue
        if token.is_keyword("JOIN"):
            in_from = True
            need_source = True
            index += 1
            continue
        if upper in _FROM_CLAUSE_KEYWORDS:
            in_from = False
            need_source = False
            index += 1
            continue
        if upper in ("INNER", "CROSS", "LEFT", "RIGHT", "FULL", "NATURAL"):
            # LEFT OUTER JOIN etc. - JOIN follows, handled above.
            index += 1
            continue
        if token.text == ",":
            if in_from:
                need_source = True
            index += 1
            continue
        if not need_source:
            if token.text == "(":
                index = _matching_paren(tokens, index) + 1
            else:
                index += 1
            continue
        if token.text == "(":
            close = _matching_paren(tokens, index)
            inner = tokens[index + 1 : close]
            if inner and inner[0].is_keyword("SELECT"):
                inner_aliases, inner_single, inner_indices = _collect_from_sources(
                    inner, table_name_folded
                )
                aliases.update(inner_aliases)
                # Inner positions are relative to the inner token list.
                source_indices.update(index + 1 + position for position in inner_indices)
                if not inner_single:
                    source_count += 1
            else:
                source_count += 1
            index = close + 1
            need_source = False
            continue
        if not _is_identifierish(token):
            index += 1
            continue
        name_start = index
        name, next_index = _qualified_name(tokens, index)
        source_count += 1
        for position in range(name_start, next_index):
            source_indices.add(position)
        if _ascii_fold(name) == table_name_folded:
            aliases.add(_ascii_fold(name))
            if next_index < len(tokens) and tokens[next_index].is_keyword("AS"):
                next_index += 1
            if (
                next_index < len(tokens)
                and tokens[next_index].kind in ("word", "identifier")
                and tokens[next_index].text.upper() not in _FROM_CLAUSE_KEYWORDS
                and not tokens[next_index].is_keyword("JOIN")
                and tokens[next_index].text != ","
            ):
                aliases.add(_ascii_fold(_unquote(tokens[next_index].text)))
                source_indices.add(next_index)
                next_index += 1
        index = next_index
        need_source = False
    single_source = source_count <= 1
    return aliases, single_source, source_indices


def plan_view(
    view_name: str,
    sql: str,
    table_name: str,
    columns: set[str],
    rename: dict[str, str],
    drop: set[str],
) -> tuple[bool, str]:
    """Analyse a view definition.

    Returns ``(touched, new_sql)``: ``touched`` is True when the view depends
    on the transformed table.  Unrelated views come back untouched.  Raises
    when a view depends on a dropped column or cannot be rewritten reliably.
    """
    try:
        parsed_name, select_index = _parse_view_header(sql)
    except ParseError as ex:
        _refuse("view", view_name, f"does not have a parseable definition ({ex})")
    if _ascii_fold(parsed_name) != _ascii_fold(view_name):
        _refuse(
            "view",
            view_name,
            f"name in its definition ({parsed_name!r}) does not match sqlite_schema",
        )
    columns_folded = _folded(columns)
    drop_folded = _folded(drop)
    rename_folded = _renamed_map(rename)
    table_name_folded = _ascii_fold(table_name)
    view_tokens = _meaningful(_lex(sql))
    # Only the SELECT body is scanned - the view name and FROM source names
    # are table names, not column references.
    references = _reference_ranges(view_tokens, select_index, len(view_tokens))
    depends_on_table = False
    aliases, single_source, source_indices = _collect_from_sources(
        view_tokens, table_name_folded
    )
    if aliases:
        depends_on_table = True
    analysis = _Analysis()
    for reference, in_subquery, token_index in references:
        if token_index in source_indices:
            continue
        folded = _ascii_fold(reference.name)
        if reference.qualifier:
            if _ascii_fold(reference.qualifier) in aliases:
                depends_on_table = True
                if folded in drop_folded:
                    _refuse(
                        "view",
                        view_name,
                        f"depends on {reference.qualifier}.{reference.name} "
                        "but that column is being dropped",
                    )
                if folded in rename_folded:
                    _edit(analysis, reference.token, rename_folded[folded])
                elif folded not in columns_folded and folded not in _PSEUDO_COLUMNS:
                    _refuse(
                        "view",
                        view_name,
                        f"references unknown column {reference.name!r}",
                    )
            continue
        if folded in drop_folded or folded in rename_folded:
            if single_source and aliases and not in_subquery:
                if folded in drop_folded:
                    _refuse(
                        "view",
                        view_name,
                        f"depends on column {reference.name!r} which is being dropped",
                    )
                _edit(analysis, reference.token, rename_folded[folded])
                continue
            _refuse(
                "view",
                view_name,
                f"references column {reference.name!r} without a table "
                "qualifier in a view whose table scope cannot be determined "
                "reliably (joins or nested subqueries)",
            )
        elif folded in columns_folded and single_source and aliases:
            depends_on_table = True
    return depends_on_table, analysis.rewritten(sql)
