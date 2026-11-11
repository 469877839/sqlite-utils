"""
Explicit column type strategies for data that arrives in several batches.

The default :func:`~sqlite_utils.utils.suggest_column_types` logic derives a
table's schema from whichever batch is imported first. When the same data
source is imported piece by piece, a later batch can disagree with that first
impression:

* measurements that looked like integers gain a fractional part, so the
  column needs to be widened to ``REAL``;
* identifier columns that looked numeric arrive with leading zeros
  (``"0012"``) or values wider than a SQLite integer, and must stay ``TEXT``;
* a column that was entirely ``NULL`` suddenly carries real values.

:class:`ColumnTypeStrategy` tracks the inferred type of every column across
batches, widens a column only when a later batch demonstrably requires it,
and records a machine-readable reason for every decision. The decision log
can be serialized with :meth:`ColumnTypeStrategy.to_log` and replayed with
:meth:`ColumnTypeStrategy.from_log` to reproduce the exact same schema and
the same accept/reject decisions, rather than only the final
``CREATE TABLE`` statement.
"""

from __future__ import annotations

import datetime
import decimal
import enum
import json
import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .utils import ANY

#: Largest integer SQLite can store in an ``INTEGER`` column.
SQLITE_MAX_INTEGER = 9223372036854775807
SQLITE_MIN_INTEGER = -9223372036854775808

#: Storage class reported by SQLite's ``typeof()`` for each column kind.
SQLITE_STORAGE_CLASS: dict[str, str] = {
    "INTEGER": "integer",
    "REAL": "real",
    "TEXT": "text",
    "BLOB": "blob",
    "ANY": "any",
}


class Semantic(enum.Enum):
    """How a column should interpret values that look numeric.

    :attr MEASUREMENT columns treat ints/floats as numbers and may be widened
    across batches. :attr IDENTIFIER columns treat every value as opaque text
    so that codes such as ``"0012"`` keep their exact spelling.
    """

    MEASUREMENT = "measurement"
    IDENTIFIER = "identifier"


class ColumnKind(enum.Enum):
    """The inferred, declared or fixed kind of a column.

    The ``value`` is the SQLite type used when the table is created or
    rebuilt.
    """

    INTEGER = "INTEGER"
    REAL = "REAL"
    TEXT = "TEXT"
    BLOB = "BLOB"
    ANY = "ANY"

    @property
    def sql_type(self) -> str:
        "SQLite type name used in ``CREATE TABLE`` / ``ALTER TABLE``."
        return str(self.value)

    @property
    def storage_class(self) -> str | None:
        "Storage class reported by ``typeof()`` for a value bound into this column."
        return SQLITE_STORAGE_CLASS.get(str(self.value))


class ValueClass(enum.Enum):
    "Classification of a single Python value before it is stored."

    NULL = "null"
    BOOL = "bool"
    INTEGER = "integer"
    BIG_INTEGER = "big_integer"
    FLOAT = "float"
    TEXT = "text"
    BLOB = "blob"
    JSON = "json"
    DATETIME = "datetime"
    DECIMAL = "decimal"
    UUID = "uuid"


#: Map of measurement value classes to the column kind they require.
MEASUREMENT_KIND: dict[ValueClass, ColumnKind] = {
    ValueClass.NULL: ColumnKind.TEXT,
    ValueClass.BOOL: ColumnKind.INTEGER,
    ValueClass.INTEGER: ColumnKind.INTEGER,
    ValueClass.BIG_INTEGER: ColumnKind.TEXT,
    ValueClass.FLOAT: ColumnKind.REAL,
    ValueClass.DECIMAL: ColumnKind.REAL,
    ValueClass.TEXT: ColumnKind.TEXT,
    ValueClass.BLOB: ColumnKind.BLOB,
    ValueClass.JSON: ColumnKind.TEXT,
    ValueClass.DATETIME: ColumnKind.TEXT,
    ValueClass.UUID: ColumnKind.TEXT,
}

#: Allowed widenings between measurement column kinds. ``None`` marks the
#: un-initialized state of a column that has only seen NULLs so far.
_MEASUREMENT_PROMOTIONS: dict[ColumnKind | None, set[ColumnKind]] = {
    None: {ColumnKind.INTEGER, ColumnKind.REAL, ColumnKind.TEXT, ColumnKind.BLOB},
    ColumnKind.INTEGER: {ColumnKind.INTEGER, ColumnKind.REAL, ColumnKind.TEXT},
    ColumnKind.REAL: {ColumnKind.REAL, ColumnKind.TEXT},
    ColumnKind.TEXT: {ColumnKind.TEXT},
    ColumnKind.BLOB: {ColumnKind.BLOB},
    ColumnKind.ANY: {ColumnKind.ANY},
}

_LEADING_ZERO_RE = re.compile(r"^[+-]?0+\d+$")
_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")
_BIGINT_OVERFLOW_TEXT = (
    "integer {value!r} is outside the signed 64-bit range SQLite stores in an "
    "INTEGER column; the column is widened to TEXT so no digits are lost"
)


def classify_value(value: Any) -> ValueClass:
    "Classify a single Python value the way the strategy will store it."
    if value is None:
        return ValueClass.NULL
    # bool must be checked before int: bool is a subclass of int
    if isinstance(value, bool):
        return ValueClass.BOOL
    if isinstance(value, int):
        if SQLITE_MIN_INTEGER <= value <= SQLITE_MAX_INTEGER:
            return ValueClass.INTEGER
        return ValueClass.BIG_INTEGER
    if isinstance(value, float):
        return ValueClass.FLOAT
    if isinstance(value, decimal.Decimal):
        return ValueClass.DECIMAL
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ValueClass.BLOB
    if isinstance(value, (dict, list, tuple)):
        return ValueClass.JSON
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return ValueClass.DATETIME
    if isinstance(value, datetime.timedelta):
        return ValueClass.TEXT
    if isinstance(value, uuid.UUID):
        return ValueClass.UUID
    if isinstance(value, str):
        return ValueClass.TEXT
    raise TypeError(f"Cannot store value of type {type(value).__name__}: {value!r}")


def looks_like_integer(text: str) -> bool:
    "True if ``text`` is an integer literal, including leading-zero codes."
    return bool(_INTEGER_RE.match(text.strip()))


def looks_like_float(text: str) -> bool:
    "True if ``text`` is a numeric literal with a fraction or exponent."
    stripped = text.strip()
    if looks_like_integer(stripped):
        return False
    return bool(_FLOAT_RE.match(stripped))


def has_leading_zero(text: str) -> bool:
    "True for digit strings whose spelling carries a leading zero, e.g. ``'0012'``."
    return bool(_LEADING_ZERO_RE.match(text.strip()))



class TypeStrategyError(Exception):
    "Base class for errors raised while applying a :class:`ColumnTypeStrategy`."


class BatchRejected(TypeStrategyError):
    """Raised when a batch cannot be accepted under the current strategy.

    Nothing from the rejected batch is inserted: the caller's table and the
    strategy state are both left untouched.
    """

    def __init__(self, violations: Sequence["TypeViolation"]):
        self.violations = list(violations)
        messages = "; ".join(str(v) for v in self.violations)
        super().__init__(
            f"Batch rejected ({len(self.violations)} violation(s)): {messages}"
        )


@dataclass(frozen=True)
class TypeViolation:
    """A single reason a value cannot fit a column's current strategy."""

    column: str
    value_class: str
    column_kind: str | None
    reason: str
    example: Any = None

    def __str__(self) -> str:
        if self.column_kind is None:
            return f"column {self.column!r}: {self.reason}"
        return f"column {self.column!r} ({self.column_kind}): {self.reason}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "value_class": self.value_class,
            "column_kind": self.column_kind,
            "reason": self.reason,
            "example": self.example,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TypeViolation":
        return cls(
            column=data["column"],
            value_class=data["value_class"],
            column_kind=data.get("column_kind"),
            reason=data["reason"],
            example=data.get("example"),
        )


@dataclass
class TypeDecision:
    """One recorded inference or promotion decision for a column.

    Decisions are append-only and fully describe *why* a column has its
    current kind, so the log can be replayed to reach the same state.
    """

    batch: int
    column: str
    action: str
    kind: str | None
    previous_kind: str | None
    reason: str
    semantic: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch": self.batch,
            "column": self.column,
            "action": self.action,
            "kind": self.kind,
            "previous_kind": self.previous_kind,
            "reason": self.reason,
            "semantic": self.semantic,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TypeDecision":
        return cls(
            batch=data["batch"],
            column=data["column"],
            action=data["action"],
            kind=data.get("kind"),
            previous_kind=data.get("previous_kind"),
            reason=data["reason"],
            semantic=data["semantic"],
            evidence=dict(data.get("evidence") or {}),
        )


@dataclass
class ColumnState:
    """Everything the strategy knows about one column."""

    name: str
    semantic: Semantic
    kind: ColumnKind | None = None
    fixed: bool = False
    seen_value_classes: set[ValueClass] = field(default_factory=set)
    observed_count: int = 0
    null_count: int = 0

    @property
    def sql_type(self) -> str | None:
        "SQLite type name for the current kind, or ``None`` while only NULLs seen."
        return None if self.kind is None else self.kind.sql_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "semantic": self.semantic.value,
            "kind": None if self.kind is None else self.kind.value,
            "fixed": self.fixed,
            "seen_value_classes": sorted(v.value for v in self.seen_value_classes),
            "observed_count": self.observed_count,
            "null_count": self.null_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ColumnState":
        return cls(
            name=data["name"],
            semantic=Semantic(data["semantic"]),
            kind=None if data.get("kind") is None else ColumnKind(data["kind"]),
            fixed=bool(data.get("fixed")),
            seen_value_classes={
                ValueClass(v) for v in data.get("seen_value_classes", [])
            },
            observed_count=int(data.get("observed_count", 0)),
            null_count=int(data.get("null_count", 0)),
        )


def _fold(name: str) -> str:
    return name.translate(
        str.maketrans(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "abcdefghijklmnopqrstuvwxyz",
        )
    )


def _resolve_casing(name: str, candidates: Iterable[str]) -> str:
    candidates = list(candidates)
    if name in candidates:
        return name
    folded = _fold(name)
    for candidate in candidates:
        if _fold(candidate) == folded:
            return candidate
    return name


def _kind_from_sql_type(sql_type: str) -> ColumnKind:
    normalized = (sql_type or "").strip().upper()
    if not normalized:
        # A typeless column has BLOB affinity in SQLite; treat it as such.
        return ColumnKind.BLOB
    if "INT" in normalized or "BOOL" in normalized:
        return ColumnKind.INTEGER
    if normalized in ("REAL", "FLOAT", "DOUBLE", "DOUBLE PRECISION", "NUMERIC"):
        return ColumnKind.REAL
    if "CHAR" in normalized or "CLOB" in normalized or "TEXT" in normalized:
        return ColumnKind.TEXT
    if normalized == "BLOB":
        return ColumnKind.BLOB
    if normalized == "ANY":
        return ColumnKind.ANY
    raise TypeStrategyError(
        f"Cannot map declared SQLite type {sql_type!r} to a column kind"
    )


def _kind_from_python(type_: Any) -> ColumnKind:
    if type_ is bool:
        return ColumnKind.INTEGER
    if type_ is int:
        return ColumnKind.INTEGER
    if type_ is float or type_ is decimal.Decimal:
        return ColumnKind.REAL
    if type_ is str or type_ in (dict, list, tuple) or type_ in (
        datetime.datetime,
        datetime.date,
        datetime.time,
        datetime.timedelta,
        uuid.UUID,
    ):
        return ColumnKind.TEXT
    if type_ in (bytes, bytearray, memoryview):
        return ColumnKind.BLOB
    if type_ is ANY:
        return ColumnKind.ANY
    if isinstance(type_, str):
        return _kind_from_sql_type(type_)
    if isinstance(type_, ColumnKind):
        return type_
    raise TypeStrategyError(f"Cannot map {type_!r} to a column kind")


class ColumnTypeStrategy:
    """Track and enforce an explicit column-type policy across batches.

    :param identifiers: column names whose values are identifiers, not
      measurements. Identifier columns are always stored as ``TEXT``, even
      when every value looks like a number, so codes such as ``"0012"`` are
      preserved exactly.
    :param fixed_types: column names pinned to a specific kind (a
      :class:`ColumnKind`, a Python type or a SQLite type string). Fixed
      columns are never widened; values that cannot fit are rejected.
    :param parse_numeric_strings: when true, string values in measurement
      columns that are exact integer/float literals are converted to numbers
      before storage. Leading-zero literals are never converted. When false
      (the default), strings always stay strings.
    """

    def __init__(
        self,
        identifiers: Iterable[str] = (),
        fixed_types: Mapping[str, Any] | None = None,
        parse_numeric_strings: bool = False,
    ) -> None:
        self._identifier_columns = {_fold(name) for name in identifiers}
        self._fixed_specs: dict[str, ColumnKind] = {}
        for name, type_ in (fixed_types or {}).items():
            self._fixed_specs[name] = _kind_from_python(type_)
        self.parse_numeric_strings = parse_numeric_strings
        self.columns: dict[str, ColumnState] = {}
        self.decisions: list[TypeDecision] = []
        self.rejections: list[TypeViolation] = []
        self.batch_count = 0
        # Kind inferred from each column in the most recently accepted batch
        self.last_batch_kinds: dict[str, str | None] = {}

    # -- introspection -----------------------------------------------------

    def semantic_for(self, column: str) -> Semantic:
        "Semantic assigned to ``column``, defaulting to :attr:`Semantic.MEASUREMENT`."
        state = self.columns.get(column)
        if state is not None:
            return state.semantic
        return (
            Semantic.IDENTIFIER
            if _fold(column) in self._identifier_columns
            else Semantic.MEASUREMENT
        )

    def kind_for(self, column: str) -> ColumnKind | None:
        "Current inferred kind for ``column``, or ``None`` if only NULLs were seen."
        state = self.columns.get(column)
        return None if state is None else state.kind

    def sql_types(self) -> dict[str, str]:
        """SQLite type mapping for all columns with a known kind.

        Columns that have only ever contained NULLs default to ``TEXT``, the
        same choice :func:`~sqlite_utils.utils.suggest_column_types` makes.
        """
        return {
            name: ("TEXT" if state.kind is None else state.kind.sql_type)
            for name, state in self.columns.items()
        }

    def state(self, column: str) -> ColumnState:
        "Return the recorded :class:`ColumnState` for ``column``."
        return self.columns[column]

    def explain(self, column: str) -> list[TypeDecision]:
        "Replayable decision history for one column, oldest first."
        return [decision for decision in self.decisions if decision.column == column]

    # -- inference ---------------------------------------------------------

    def infer_batch(
        self, records: Sequence[Mapping[str, Any]]
    ) -> dict[str, str | None]:
        """Validate a batch against the strategy, widening columns if allowed.

        On success the strategy state is updated, every change is appended to
        :attr:`decisions`, and the per-column kinds inferred for this batch
        are returned. On failure :class:`BatchRejected` is raised and no state
        changes are kept, so a rejected batch cannot leave half-imported
        types behind.

        :param records: the batch's records (dictionaries).
        """
        batch_no = self.batch_count + 1
        # Work on a private copy so a rejection rolls everything back.
        draft = {name: _clone_state(state) for name, state in self.columns.items()}
        draft_decisions: list[TypeDecision] = []
        violations: list[TypeViolation] = []
        batch_kinds: dict[str, str | None] = {}

        def state_for(column: str) -> ColumnState:
            if column not in draft:
                fixed_kind = self._fixed_specs.get(
                    _resolve_casing(column, self._fixed_specs)
                )
                semantic = (
                    Semantic.IDENTIFIER
                    if _fold(column) in self._identifier_columns
                    else Semantic.MEASUREMENT
                )
                draft[column] = ColumnState(
                    name=column,
                    semantic=semantic,
                    kind=fixed_kind,
                    fixed=fixed_kind is not None,
                )
                if fixed_kind is not None:
                    draft_decisions.append(
                        TypeDecision(
                            batch=batch_no,
                            column=column,
                            action="fix",
                            kind=fixed_kind.value,
                            previous_kind=None,
                            reason=f"caller fixed column to {fixed_kind.sql_type}",
                            semantic=semantic.value,
                        )
                    )
            return draft[column]

        for record_index, record in enumerate(records):
            for column, raw_value in record.items():
                state = state_for(column)
                value_class = classify_value(raw_value)
                evidence = {
                    "value_class": value_class.value,
                    "record_index": record_index,
                }
                if value_class is ValueClass.NULL:
                    state.null_count += 1
                    state.observed_count += 1
                    continue
                state.observed_count += 1
                state.seen_value_classes.add(value_class)
                precheck = self._precheck_value(state, raw_value, value_class)
                if precheck is not None:
                    violations.append(precheck)
                    continue
                natural, reason = self._required_kind(state, raw_value, value_class)
                evidence["required_kind"] = natural.value
                batch_kinds[column] = natural.value
                if state.kind is None:
                    state.kind = natural
                    draft_decisions.append(
                        TypeDecision(
                            batch=batch_no,
                            column=column,
                            action="infer",
                            kind=natural.value,
                            previous_kind=None,
                            reason=reason,
                            semantic=state.semantic.value,
                            evidence=evidence,
                        )
                    )
                    continue
                if natural == state.kind:
                    draft_decisions.append(
                        TypeDecision(
                            batch=batch_no,
                            column=column,
                            action="accept",
                            kind=state.kind.value,
                            previous_kind=state.kind.value,
                            reason=reason,
                            semantic=state.semantic.value,
                            evidence=evidence,
                        )
                    )
                    continue
                fits = self._value_fits_kind(raw_value, value_class, state.kind)
                if fits:
                    # e.g. an integer landing in a column already widened to REAL
                    draft_decisions.append(
                        TypeDecision(
                            batch=batch_no,
                            column=column,
                            action="accept",
                            kind=state.kind.value,
                            previous_kind=state.kind.value,
                            reason=(
                                f"{value_class.value} value fits the existing "
                                f"{state.kind.sql_type} column without a type change"
                            ),
                            semantic=state.semantic.value,
                            evidence=evidence,
                        )
                    )
                    continue
                allowed = _MEASUREMENT_PROMOTIONS.get(state.kind, set())
                if state.fixed or state.semantic is Semantic.IDENTIFIER:
                    allowed = {state.kind}
                if natural not in allowed:
                    violations.append(
                        self._fit_violation(state, value_class, reason, raw_value)
                    )
                    continue
                previous_kind = state.kind
                state.kind = natural
                draft_decisions.append(
                    TypeDecision(
                        batch=batch_no,
                        column=column,
                        action="promote",
                        kind=natural.value,
                        previous_kind=previous_kind.value,
                        reason=reason,
                        semantic=state.semantic.value,
                        evidence=evidence,
                    )
                )

        # Columns present in the batch but only as NULL stay untyped for now.
        for column in draft:
            batch_kinds.setdefault(
                column,
                None if draft[column].kind is None else draft[column].kind.value,
            )

        if violations:
            self.rejections.extend(violations)
            raise BatchRejected(violations)

        self.columns = draft
        self.decisions.extend(draft_decisions)
        self.last_batch_kinds = batch_kinds
        self.batch_count = batch_no
        return batch_kinds

    def _required_kind(
        self,
        state: ColumnState,
        value: Any,
        value_class: ValueClass,
    ) -> tuple[ColumnKind, str]:
        """Kind required to store ``value`` plus a human-readable reason."""
        if state.semantic is Semantic.IDENTIFIER:
            reason = self._identifier_reason(value, value_class)
            return ColumnKind.TEXT, reason
        if value_class is ValueClass.BIG_INTEGER:
            return ColumnKind.TEXT, _BIGINT_OVERFLOW_TEXT.format(value=value)
        if value_class is ValueClass.TEXT and self.parse_numeric_strings:
            numeric_kind = self._numeric_literal_kind(value)
            if numeric_kind is not None:
                return numeric_kind, (
                    f"string {value!r} is an exact {numeric_kind.sql_type.lower()} "
                    "literal and parse_numeric_strings is enabled"
                )
        kind = MEASUREMENT_KIND[value_class]
        return kind, self._measurement_reason(value_class, value)

    @staticmethod
    def _identifier_reason(value: Any, value_class: ValueClass) -> str:
        if value_class is ValueClass.TEXT and has_leading_zero(value):
            return (
                f"identifier {value!r} keeps its leading zeros; storing it as a "
                "number would rewrite the code"
            )
        if value_class is ValueClass.BIG_INTEGER:
            return (
                f"identifier {value!r} is wider than a SQLite integer; it is stored "
                "as text so every digit is preserved"
            )
        if value_class is ValueClass.TEXT:
            return "identifier columns always store values as text, never as numbers"
        return (
            "identifier values are stored as text; "
            f"{value_class.value} value is serialized to text"
        )

    @staticmethod
    def _measurement_reason(value_class: ValueClass, value: Any) -> str:
        if value_class is ValueClass.FLOAT:
            return "fractional measurement requires REAL; INTEGER would truncate it"
        if value_class is ValueClass.DECIMAL:
            return "Decimal measurement is stored as REAL"
        if value_class is ValueClass.BOOL:
            return "boolean is stored as 0/1 in an INTEGER column"
        if value_class is ValueClass.INTEGER:
            return "whole-number measurement fits an INTEGER column"
        if value_class is ValueClass.BLOB:
            return "binary value requires a BLOB column"
        if value_class in (ValueClass.JSON, ValueClass.DATETIME, ValueClass.UUID):
            return f"{value_class.value} value is serialized to text"
        if value_class is ValueClass.TEXT:
            if has_leading_zero(value):
                return (
                    f"digit string {value!r} carries a leading zero, so it is not "
                    "converted to a number"
                )
            return "string value is stored as text"
        return f"{value_class.value} value is stored as text"

    @staticmethod
    def _numeric_literal_kind(text: str) -> ColumnKind | None:
        stripped = text.strip()
        if has_leading_zero(stripped):
            return None
        if looks_like_integer(stripped):
            try:
                integer = int(stripped)
            except ValueError:
                return None
            if SQLITE_MIN_INTEGER <= integer <= SQLITE_MAX_INTEGER:
                return ColumnKind.INTEGER
            return ColumnKind.TEXT
        if looks_like_float(stripped):
            return ColumnKind.REAL
        return None

    def _precheck_value(
        self,
        state: ColumnState,
        value: Any,
        value_class: ValueClass,
    ) -> TypeViolation | None:
        "Hard refusals that apply regardless of the column's current kind."
        if value_class is ValueClass.FLOAT and not math.isfinite(value):
            return TypeViolation(
                column=state.name,
                value_class=value_class.value,
                column_kind=None if state.kind is None else state.kind.value,
                reason=(
                    f"float {value!r} is not finite; SQLite would store it as NULL, "
                    "so the batch is rejected instead"
                ),
                example=_safe_example(value),
            )
        if (
            value_class is ValueClass.BLOB
            and state.semantic is Semantic.IDENTIFIER
            and state.kind is not ColumnKind.BLOB
        ):
            return TypeViolation(
                column=state.name,
                value_class=value_class.value,
                column_kind=None if state.kind is None else state.kind.value,
                reason=(
                    "identifier column stores text; a binary value cannot be "
                    "serialized without loss"
                ),
                example=_safe_example(value),
            )
        return None

    def _value_fits_kind(
        self, value: Any, value_class: ValueClass, kind: ColumnKind
    ) -> bool:
        """Whether ``value`` can be stored in a column of ``kind`` without loss.

        This models SQLite's value/affinity rules plus the extra guarantee that
        a digit string is never silently rewritten as a number.
        """
        if kind is ColumnKind.ANY:
            return True
        if kind is ColumnKind.INTEGER:
            if value_class in (ValueClass.INTEGER, ValueClass.BOOL):
                return True
            if value_class is ValueClass.TEXT and self.parse_numeric_strings:
                return self._numeric_literal_kind(value) is ColumnKind.INTEGER
            return False
        if kind is ColumnKind.REAL:
            if value_class in (
                ValueClass.INTEGER,
                ValueClass.BOOL,
                ValueClass.FLOAT,
                ValueClass.DECIMAL,
            ):
                return True
            if value_class is ValueClass.TEXT and self.parse_numeric_strings:
                return self._numeric_literal_kind(value) in (
                    ColumnKind.INTEGER,
                    ColumnKind.REAL,
                )
            return False
        if kind is ColumnKind.TEXT:
            # Numeric and serialized values have an exact text representation.
            return value_class is not ValueClass.BLOB
        if kind is ColumnKind.BLOB:
            return value_class is ValueClass.BLOB
        return False

    def _fit_violation(
        self,
        state: ColumnState,
        value_class: ValueClass,
        reason: str,
        example: Any,
    ) -> TypeViolation:
        if state.fixed:
            reason = (
                f"column is fixed to {state.kind.sql_type} and cannot be widened; "
                f"{reason}"
            )
        elif state.semantic is Semantic.IDENTIFIER:
            reason = (
                f"identifier column is {state.kind.sql_type}; {reason}"
            )
        return TypeViolation(
            column=state.name,
            value_class=value_class.value,
            column_kind=state.kind.value,
            reason=reason,
            example=_safe_example(example),
        )

    # -- value coercion ----------------------------------------------------

    def prepare_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Convert a record's values to SQLite-bindable values for the current kinds.

        The record's keys must already be known to the strategy (call
        :meth:`infer_batch` first); unknown columns raise
        :class:`TypeStrategyError`.
        """
        prepared: dict[str, Any] = {}
        for column, value in record.items():
            if column not in self.columns:
                raise TypeStrategyError(
                    f"Column {column!r} was not seen by infer_batch(); "
                    "run infer_batch() before prepare_record()"
                )
            prepared[column] = self.prepare_value(column, value)
        return prepared

    def prepare_value(self, column: str, value: Any) -> Any:
        "Convert a single value for storage in ``column`` under the current kind."
        state = self.columns[column]
        kind = state.kind
        if value is None:
            return None
        value_class = classify_value(value)
        if state.semantic is Semantic.IDENTIFIER and not state.fixed:
            return self._to_text(value, value_class)
        if kind is ColumnKind.ANY:
            return self._default_serialize(value, value_class)
        if kind is ColumnKind.INTEGER:
            if value_class is ValueClass.BOOL:
                return int(value)
            if value_class is ValueClass.INTEGER:
                return value
            if value_class is ValueClass.TEXT and self.parse_numeric_strings:
                parsed = self._numeric_literal_kind(value)
                if parsed is ColumnKind.INTEGER:
                    return int(value.strip())
            return self._default_serialize(value, value_class)
        if kind is ColumnKind.REAL:
            if value_class is ValueClass.BOOL:
                return float(value)
            if value_class is ValueClass.INTEGER:
                return float(value)
            if value_class is ValueClass.FLOAT:
                return value
            if value_class is ValueClass.DECIMAL:
                return float(value)
            if value_class is ValueClass.TEXT and self.parse_numeric_strings:
                parsed = self._numeric_literal_kind(value)
                if parsed is ColumnKind.INTEGER:
                    return float(int(value.strip()))
                if parsed is ColumnKind.REAL:
                    return float(value.strip())
            return self._default_serialize(value, value_class)
        if kind is ColumnKind.TEXT:
            if value_class is ValueClass.TEXT:
                return value
            if value_class is ValueClass.BIG_INTEGER:
                return str(value)
            if value_class is ValueClass.INTEGER:
                return str(value)
            if value_class is ValueClass.BOOL:
                return "1" if value else "0"
            if value_class is ValueClass.FLOAT:
                return repr(value)
            return self._default_serialize(value, value_class)
        if kind is ColumnKind.BLOB:
            if value_class is ValueClass.BLOB:
                return bytes(value)
            return self._default_serialize(value, value_class)
        return self._default_serialize(value, value_class)

    @staticmethod
    def _to_text(value: Any, value_class: ValueClass) -> str:
        if value_class is ValueClass.TEXT:
            return value
        if value_class is ValueClass.BOOL:
            return "1" if value else "0"
        if value_class is ValueClass.BIG_INTEGER:
            return str(value)
        if value_class is ValueClass.INTEGER:
            return str(value)
        if value_class is ValueClass.FLOAT:
            return repr(value)
        return ColumnTypeStrategy._default_serialize(value, value_class)

    @staticmethod
    def _default_serialize(value: Any, value_class: ValueClass) -> Any:
        "Match the serialization sqlite-utils uses for these Python types."
        if value_class is ValueClass.DECIMAL:
            return float(value)
        if value_class is ValueClass.JSON:
            return json.dumps(value, default=repr, ensure_ascii=False)
        if value_class is ValueClass.DATETIME:
            return value.isoformat()
        if value_class is ValueClass.UUID:
            return str(value)
        if value_class is ValueClass.BLOB:
            return bytes(value)
        if value_class is ValueClass.TEXT and isinstance(value, datetime.timedelta):
            return str(value)
        return value

    # -- decision log ------------------------------------------------------

    def to_log(self) -> list[dict[str, Any]]:
        """Serialize the full decision log as JSON-compatible dictionaries.

        Replaying these entries with :meth:`from_log` reproduces the same
        column kinds and the same accept/reject outcomes, not just the final
        ``CREATE TABLE`` statement.
        """
        return [decision.to_dict() for decision in self.decisions]

    def to_json(self) -> str:
        "Serialize the complete strategy state (columns, decisions, rejections)."
        return json.dumps(
            {
                "columns": [state.to_dict() for state in self.columns.values()],
                "decisions": self.to_log(),
                "rejections": [violation.to_dict() for violation in self.rejections],
                "batch_count": self.batch_count,
                "parse_numeric_strings": self.parse_numeric_strings,
            },
            indent=2,
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, payload: str) -> "ColumnTypeStrategy":
        "Restore a strategy previously serialized with :meth:`to_json`."
        data = json.loads(payload)
        strategy = cls(parse_numeric_strings=bool(data.get("parse_numeric_strings")))
        strategy.columns = {
            column["name"]: ColumnState.from_dict(column) for column in data["columns"]
        }
        strategy.decisions = [TypeDecision.from_dict(d) for d in data["decisions"]]
        strategy.rejections = [TypeViolation.from_dict(v) for v in data["rejections"]]
        strategy.batch_count = int(data["batch_count"])
        return strategy

    @classmethod
    def from_log(
        cls,
        log: Sequence[Mapping[str, Any]],
        *,
        identifiers: Iterable[str] = (),
        fixed_types: Mapping[str, Any] | None = None,
        parse_numeric_strings: bool = False,
    ) -> "ColumnTypeStrategy":
        """Rebuild a strategy by replaying a decision log.

        Only ``infer``, ``promote`` and ``fix`` entries change column kinds;
        ``accept`` entries are validated against the state they describe so a
        tampered or truncated log fails loudly instead of silently producing a
        different schema.
        """
        strategy = cls(
            identifiers=identifiers,
            fixed_types=fixed_types,
            parse_numeric_strings=parse_numeric_strings,
        )
        for entry in log:
            decision = TypeDecision.from_dict(entry)
            column = decision.column
            state = strategy.columns.get(column)
            if state is None:
                state = ColumnState(
                    name=column,
                    semantic=Semantic(decision.semantic),
                )
                strategy.columns[column] = state
            previous = None if state.kind is None else state.kind.value
            if previous != decision.previous_kind:
                raise TypeStrategyError(
                    f"Decision log is inconsistent at batch {decision.batch} for "
                    f"column {column!r}: expected previous kind "
                    f"{decision.previous_kind!r} but replay produced {previous!r}"
                )
            if decision.action in ("infer", "promote", "fix"):
                state.kind = ColumnKind(decision.kind)
                state.fixed = decision.action == "fix"
            strategy.decisions.append(decision)
            strategy.batch_count = max(strategy.batch_count, decision.batch)
        return strategy

    # -- table application -------------------------------------------------

    def check_migration(
        self, table: Any, new_types: Mapping[str, ColumnKind]
    ) -> list[TypeViolation]:
        """Check whether existing rows survive a rebuild under ``new_types``.

        Every existing value is read with its real SQLite storage class and
        checked for a lossless conversion to the target kind. The returned
        list is empty when the migration is safe.
        """
        # Imported lazily so importing this module never pulls in sqlite_utils.db.
        from .db import quote_identifier

        violations: list[TypeViolation] = []
        existing = {column.name: column.type for column in table.columns}
        targets = {
            _resolve_casing(name, existing): kind for name, kind in new_types.items()
        }
        if not existing:
            return violations
        select_parts = []
        for name in existing:
            select_parts.append(quote_identifier(name))
            select_parts.append(f"typeof({quote_identifier(name)})")
        rows = table.db.execute(
            f"SELECT {', '.join(select_parts)} FROM {quote_identifier(table.name)}"
        ).fetchall()
        for row in rows:
            for index, name in enumerate(existing):
                value = row[index * 2]
                storage_class = row[index * 2 + 1]
                target = targets.get(name)
                if target is None or value is None:
                    continue
                problem = _lossless_problem(storage_class, value, target)
                if problem is not None:
                    violations.append(
                        TypeViolation(
                            column=name,
                            value_class=storage_class,
                            column_kind=target.value,
                            reason=problem,
                            example=_safe_example(value),
                        )
                    )
        return violations

    def apply_to_table(
        self,
        table: Any,
        records: Sequence[Mapping[str, Any]],
        *,
        strict: bool = False,
    ) -> Any:
        """Infer a batch and atomically write it to ``table``.

        The whole operation - inference, any table rebuild and the insert -
        is wrapped in one transaction, so a rejected batch never leaves the
        table half-imported. New columns are added for keys the table did not
        have; existing columns are rebuilt when a batch requires a wider
        kind, but only after :meth:`check_migration` proves every old row
        converts without loss.

        :param table: a :class:`sqlite_utils.db.Table`.
        :param records: the batch's records.
        :param strict: create new columns/tables in STRICT mode.
        """
        if not records:
            return table
        snapshot = self._snapshot()
        try:
            self.infer_batch(records)
            prepared = [self.prepare_record(record) for record in records]
            self._apply_prepared(table, prepared, strict)
        except Exception:
            # The batch was rejected: roll back both the database transaction
            # (handled by atomic()) and the strategy's inferred state, so the
            # caller can fix the data and resubmit the same batch cleanly.
            self._restore(snapshot)
            raise
        return table

    def _snapshot(self) -> tuple[Any, ...]:
        return (
            {name: _clone_state(state) for name, state in self.columns.items()},
            len(self.decisions),
            self.batch_count,
            dict(self.last_batch_kinds),
        )

    def _restore(self, snapshot: tuple[Any, ...]) -> None:
        columns, decision_count, batch_count, last_batch_kinds = snapshot
        self.columns = columns
        del self.decisions[decision_count:]
        self.batch_count = batch_count
        self.last_batch_kinds = dict(last_batch_kinds)

    def _apply_prepared(
        self, table: Any, prepared: Sequence[Mapping[str, Any]], strict: bool
    ) -> None:
        with table.db.atomic():
            if not table.exists():
                table.create(self.sql_types(), strict=strict)
            else:
                existing_types = {
                    column.name: _kind_from_sql_type(column.type)
                    for column in table.columns
                }
                missing: dict[str, ColumnKind] = {}
                promoted: dict[str, ColumnKind] = {}
                for name, state in self.columns.items():
                    actual = _resolve_casing(name, existing_types)
                    if actual not in existing_types:
                        missing[actual] = state.kind or ColumnKind.TEXT
                    elif (
                        state.kind is not None
                        and existing_types[actual] != state.kind
                    ):
                        promoted[actual] = state.kind
                for name, kind in missing.items():
                    table.add_column(name, kind.sql_type)
                if promoted:
                    violations = self.check_migration(table, promoted)
                    if violations:
                        self.rejections.extend(violations)
                        raise BatchRejected(violations)
                    table.transform(
                        types={name: kind.sql_type for name, kind in promoted.items()}
                    )
            table.insert_all(prepared, alter=True)


def _lossless_problem(storage_class: str, value: Any, target: ColumnKind) -> str | None:
    "Return a reason string if converting ``value`` to ``target`` would lose data."
    if target is ColumnKind.ANY:
        return None
    if storage_class == "null":
        return None
    if target is ColumnKind.BLOB:
        if storage_class == "blob":
            return None
        return (
            f"existing {storage_class} value cannot be stored in BLOB "
            "without encoding"
        )
    if target is ColumnKind.TEXT:
        # Integers and floats each have an exact text rendering, so widening a
        # numeric column to TEXT (the only option for values that do not fit a
        # SQLite number, e.g. very large integers) preserves every digit.
        if storage_class in ("text", "integer", "real"):
            return None
        return (
            f"existing {storage_class} value cannot be rendered as text without "
            "encoding it first"
        )
    if storage_class == "text":
        if target is ColumnKind.INTEGER:
            text = str(value).strip()
            if has_leading_zero(text):
                return (
                    f"text {value!r} carries a leading zero; INTEGER would rewrite it"
                )
            if looks_like_integer(text):
                try:
                    integer = int(text)
                except ValueError:
                    return f"text {value!r} is not a valid integer"
                if SQLITE_MIN_INTEGER <= integer <= SQLITE_MAX_INTEGER:
                    return None
            return f"text {value!r} cannot be converted to INTEGER without rewriting it"
        if target is ColumnKind.REAL:
            text = str(value).strip()
            if looks_like_float(text) or looks_like_integer(text):
                try:
                    float(text)
                except ValueError:
                    return f"text {value!r} is not a valid number"
                if has_leading_zero(text):
                    return f"text {value!r} carries a leading zero"
                return None
            return f"text {value!r} cannot be converted to REAL without rewriting it"
        return f"text value cannot be stored in {target.sql_type}"
    if storage_class == "integer":
        if target is ColumnKind.INTEGER:
            return None
        if target is ColumnKind.REAL:
            # Every 64-bit integer is exactly representable as a float once it
            # is within 2^53; larger integers lose their low bits.
            if abs(int(value)) <= 2**53:
                return None
            return f"integer {value} would lose precision when stored as REAL"
        return f"integer value cannot be stored in {target.sql_type}"
    if storage_class == "real":
        if target is ColumnKind.REAL:
            return None
        if target is ColumnKind.INTEGER:
            return (
                f"real value {value} has a fractional part that INTEGER "
                "would truncate"
            )
        return f"real value cannot be stored in {target.sql_type}"
    if storage_class == "blob":
        return f"existing BLOB value cannot be stored in {target.sql_type}"
    return f"unsupported storage class {storage_class!r}"


def _safe_example(value: Any, limit: int = 128) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(value))} bytes>"
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "..."
    return value


def _clone_state(state: ColumnState) -> ColumnState:
    return ColumnState(
        name=state.name,
        semantic=state.semantic,
        kind=state.kind,
        fixed=state.fixed,
        seen_value_classes=set(state.seen_value_classes),
        observed_count=state.observed_count,
        null_count=state.null_count,
    )
