"""Tests for sqlite_utils.type_strategy."""

import datetime
import decimal
import uuid

import pytest

from sqlite_utils import Database
from sqlite_utils.type_strategy import (
    BatchRejected,
    ColumnKind,
    ColumnTypeStrategy,
    Semantic,
    TypeStrategyError,
    ValueClass,
    classify_value,
    has_leading_zero,
    looks_like_float,
    looks_like_integer,
)


@pytest.fixture
def db():
    return Database(memory=True)


def storage_classes(database, table, *columns):
    quoted = ", ".join(f'typeof("{c}")' for c in columns)
    return database.execute(f"SELECT {quoted} FROM {table}").fetchall()


def test_classify_value_basic():
    assert classify_value(None) is ValueClass.NULL
    assert classify_value(True) is ValueClass.BOOL
    assert classify_value(3) is ValueClass.INTEGER
    assert classify_value(2**63) is ValueClass.BIG_INTEGER
    assert classify_value(1.5) is ValueClass.FLOAT
    assert classify_value(decimal.Decimal("1.5")) is ValueClass.DECIMAL
    assert classify_value("x") is ValueClass.TEXT
    assert classify_value(b"x") is ValueClass.BLOB
    assert classify_value(bytearray(b"x")) is ValueClass.BLOB
    assert classify_value({"a": 1}) is ValueClass.JSON
    assert classify_value([1, 2]) is ValueClass.JSON
    assert classify_value(datetime.date(2026, 1, 1)) is ValueClass.DATETIME
    assert classify_value(uuid.uuid4()) is ValueClass.UUID
    with pytest.raises(TypeError):
        classify_value(object())


def test_literal_helpers():
    assert looks_like_integer("0012")
    assert looks_like_integer("-7")
    assert not looks_like_integer("1.5")
    assert looks_like_float("1.5")
    assert looks_like_float("1e3")
    assert not looks_like_float("12")
    assert has_leading_zero("0012")
    assert has_leading_zero("-007")
    assert not has_leading_zero("12")
    assert not has_leading_zero("0")



def test_integer_then_decimal_promotes_to_real(db):
    strategy = ColumnTypeStrategy()
    table = db["readings"]
    strategy.apply_to_table(table, [{"temp": 21, "station": "a"}])
    assert strategy.kind_for("temp") is ColumnKind.INTEGER
    assert storage_classes(db, '"readings"', "temp") == [("integer",)]

    strategy.apply_to_table(table, [{"temp": 21.5, "station": "b"}])
    assert strategy.kind_for("temp") is ColumnKind.REAL
    rows = db.execute(
        'SELECT temp, typeof(temp) FROM "readings" ORDER BY station'
    ).fetchall()
    assert rows == [(21.0, "real"), (21.5, "real")]

    promotion = next(d for d in strategy.explain("temp") if d.action == "promote")
    assert promotion.previous_kind == "INTEGER"
    assert promotion.kind == "REAL"
    assert "truncate" in promotion.reason


def test_identifier_keeps_leading_zero_and_refuses_numeric_storage(db):
    strategy = ColumnTypeStrategy(identifiers=["code"])
    table = db["devices"]
    strategy.apply_to_table(table, [{"code": "0012", "value": 1}])
    # A later batch arriving with plain digit strings must not widen or convert
    strategy.apply_to_table(table, [{"code": "7", "value": 2}])
    rows = db.execute(
        "SELECT code, typeof(code) FROM devices ORDER BY value"
    ).fetchall()
    assert rows == [("0012", "text"), ("7", "text")]
    assert strategy.kind_for("code") is ColumnKind.TEXT
    assert strategy.semantic_for("code") is Semantic.IDENTIFIER
    decision = strategy.explain("code")[0]
    assert "leading zero" in decision.reason


def test_measurement_digit_string_is_not_converted_by_default(db):
    strategy = ColumnTypeStrategy()
    table = db["m"]
    strategy.apply_to_table(table, [{"code": "12"}])
    assert storage_classes(db, "m", "code") == [("text",)]
    # "0012" in a measurement column is text too
    strategy.apply_to_table(table, [{"code": "0012"}])
    assert [r["code"] for r in table.rows] == ["12", "0012"]


def test_parse_numeric_strings_converts_plain_literals_only(db):
    strategy = ColumnTypeStrategy(parse_numeric_strings=True)
    table = db["m"]
    strategy.infer_batch([{"a": "12", "b": "0012", "c": "1.5", "d": "x"}])
    assert strategy.kind_for("a") is ColumnKind.INTEGER
    assert strategy.kind_for("b") is ColumnKind.TEXT
    assert strategy.kind_for("c") is ColumnKind.REAL
    assert strategy.kind_for("d") is ColumnKind.TEXT
    strategy.apply_to_table(table, [{"a": "12", "b": "0012", "c": "1.5", "d": "x"}])
    assert storage_classes(db, "m", "a", "b", "c", "d") == [
        ("integer", "text", "real", "text")
    ]
    assert [r["b"] for r in table.rows] == ["0012"]


def test_fixed_column_is_never_widened(db):
    strategy = ColumnTypeStrategy(fixed_types={"count": int})
    table = db["m"]
    strategy.apply_to_table(table, [{"count": 5}])
    assert strategy.state("count").fixed is True
    with pytest.raises(BatchRejected) as excinfo:
        strategy.apply_to_table(table, [{"count": 1.25}])
    assert excinfo.value.violations[0].column == "count"
    assert "fixed to INTEGER" in excinfo.value.violations[0].reason
    # Nothing was imported and the kind/table are untouched
    assert strategy.kind_for("count") is ColumnKind.INTEGER
    assert [r["count"] for r in table.rows] == [5]


def test_fixed_text_column_stringifies_numbers(db):
    strategy = ColumnTypeStrategy(fixed_types={"id": str})
    table = db["m"]
    strategy.apply_to_table(table, [{"id": 42}])
    assert storage_classes(db, "m", "id") == [("text",)]
    assert list(table.rows)[0]["id"] == "42"


def test_null_only_column_defaults_to_text_then_takes_real_type(db):
    strategy = ColumnTypeStrategy()
    table = db["m"]
    strategy.apply_to_table(table, [{"x": None}])
    assert strategy.kind_for("x") is None
    assert strategy.sql_types()["x"] == "TEXT"
    strategy.apply_to_table(table, [{"x": 3.0}])
    assert strategy.kind_for("x") is ColumnKind.REAL
    assert storage_classes(db, "m", "x") == [("null",), ("real",)]


def test_missing_column_in_later_batch_is_added(db):
    strategy = ColumnTypeStrategy()
    table = db["m"]
    strategy.apply_to_table(table, [{"a": 1}])
    strategy.apply_to_table(table, [{"a": 2, "b": "late"}])
    assert [c.name for c in table.columns] == ["a", "b"]
    assert [dict(r) for r in table.rows] == [
        {"a": 1, "b": None},
        {"a": 2, "b": "late"},
    ]


def test_blob_column_and_rejected_text(db):
    strategy = ColumnTypeStrategy()
    table = db["m"]
    strategy.apply_to_table(table, [{"payload": b"\x00\x01"}])
    assert strategy.kind_for("payload") is ColumnKind.BLOB
    assert storage_classes(db, "m", "payload") == [("blob",)]
    with pytest.raises(BatchRejected):
        strategy.apply_to_table(table, [{"payload": "plain text"}])
    assert table.count == 1


def test_identifier_rejects_binary_values(db):
    strategy = ColumnTypeStrategy(identifiers=["code"])
    with pytest.raises(BatchRejected) as excinfo:
        strategy.infer_batch([{"code": b"\x00"}])
    assert "binary" in excinfo.value.violations[0].reason


def test_non_finite_floats_are_rejected(db):
    strategy = ColumnTypeStrategy()
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(BatchRejected):
            strategy.infer_batch([{"x": bad}])
        assert strategy.batch_count == 0


def test_big_integer_is_stored_as_exact_text(db):
    strategy = ColumnTypeStrategy()
    table = db["m"]
    huge = 10**30
    strategy.apply_to_table(table, [{"x": huge}])
    assert strategy.kind_for("x") is ColumnKind.TEXT
    value, storage = db.execute("SELECT x, typeof(x) FROM m").fetchone()
    assert value == str(huge)
    assert storage == "text"
    assert list(table.rows)[0]["x"] == str(huge)


def test_big_integer_identifier_reads_back_unchanged(db):
    strategy = ColumnTypeStrategy(identifiers=["code"])
    table = db["m"]
    strategy.apply_to_table(table, [{"code": 2**63}])
    assert list(table.rows)[0]["code"] == str(2**63)
    assert storage_classes(db, "m", "code") == [("text",)]


def test_boolean_decimal_and_json_serialization(db):
    strategy = ColumnTypeStrategy()
    table = db["m"]
    strategy.apply_to_table(
        table,
        [
            {
                "flag": True,
                "amount": decimal.Decimal("2.50"),
                "meta": {"k": [1, 2]},
                "when": datetime.datetime(2026, 1, 1, 12, 0, 0),
            }
        ],
    )
    row = list(table.rows)[0]
    assert row["flag"] == 1
    assert row["amount"] == 2.5
    assert row["meta"] == '{"k": [1, 2]}'
    assert row["when"] == "2026-01-01T12:00:00"
    assert storage_classes(db, "m", "flag", "amount", "meta", "when") == [
        ("integer", "real", "text", "text")
    ]


def test_rejected_batch_does_not_import_half_its_rows(db):
    strategy = ColumnTypeStrategy(fixed_types={"x": int})
    table = db["m"]
    strategy.apply_to_table(table, [{"x": 1}])
    bad_batch = [{"x": 2}, {"x": 3.5}, {"x": 4}]
    with pytest.raises(BatchRejected):
        strategy.apply_to_table(table, bad_batch)
    assert table.count == 1
    assert strategy.batch_count == 1
    assert strategy.kind_for("x") is ColumnKind.INTEGER
    # The fixed-up batch can be resubmitted cleanly
    strategy.apply_to_table(table, [{"x": 2}, {"x": 3}, {"x": 4}])
    assert table.count == 4


def test_nan_mid_batch_rolls_back_every_row(db):
    strategy = ColumnTypeStrategy()
    table = db["m"]
    strategy.apply_to_table(table, [{"x": 1}])
    with pytest.raises(BatchRejected):
        strategy.apply_to_table(table, [{"x": 2}, {"x": float("nan")}, {"x": 4}])
    assert table.count == 1


def test_migration_check_blocks_precision_loss(db):
    strategy = ColumnTypeStrategy()
    table = db["m"]
    big = 2**53 + 1
    strategy.apply_to_table(table, [{"x": big}])
    with pytest.raises(BatchRejected) as excinfo:
        strategy.apply_to_table(table, [{"x": 1.5}])
    assert "lose precision" in excinfo.value.violations[0].reason
    # Old value and INTEGER schema survive
    assert list(table.rows)[0]["x"] == big
    assert [c.type for c in table.columns] == ["INTEGER"]


def test_migration_check_blocks_leading_zero_text_to_integer(db):
    table = db["m"]
    table.insert({"code": "0012"})  # legacy TEXT data
    strategy = ColumnTypeStrategy()
    strategy.infer_batch([{"code": "0012"}])
    violations = strategy.check_migration(
        table, {"code": ColumnKind.INTEGER}
    )
    assert violations
    assert "leading zero" in violations[0].reason


def test_integer_to_real_migration_is_allowed_for_small_values(db):
    strategy = ColumnTypeStrategy()
    table = db["m"]
    strategy.apply_to_table(table, [{"x": 100}])
    strategy.apply_to_table(table, [{"x": 2.5}])
    assert [r["x"] for r in table.rows] == [100.0, 2.5]


def test_strict_table_creation(db):
    strategy = ColumnTypeStrategy(identifiers=["code"])
    table = db["m"]
    strategy.apply_to_table(table, [{"code": "0012", "v": 1}], strict=True)
    assert "STRICT" in table.schema


def test_decision_log_replays_to_identical_state(db):
    strategy = ColumnTypeStrategy(identifiers=["code"])
    table = db["m"]
    strategy.apply_to_table(table, [{"code": "0012", "temp": 21}])
    strategy.apply_to_table(table, [{"code": "7", "temp": 2.5}])
    log = strategy.to_log()

    replayed = ColumnTypeStrategy.from_log(log, identifiers=["code"])
    assert replayed.sql_types() == strategy.sql_types()
    assert [d.to_dict() for d in replayed.decisions] == log
    assert replayed.batch_count == 2


def test_decision_log_json_roundtrip():
    strategy = ColumnTypeStrategy(identifiers=["code"])
    strategy.infer_batch([{"code": "0012", "temp": 21}])
    restored = ColumnTypeStrategy.from_json(strategy.to_json())
    assert restored.sql_types() == strategy.sql_types()
    assert restored.explain("code")[0].reason == strategy.explain("code")[0].reason


def test_tampered_log_is_detected():
    strategy = ColumnTypeStrategy(identifiers=["code"])
    strategy.infer_batch([{"code": "0012"}])
    log = strategy.to_log()
    log[0]["previous_kind"] = "REAL"
    with pytest.raises(TypeStrategyError):
        ColumnTypeStrategy.from_log(log, identifiers=["code"])


def test_prepare_record_requires_inference_first():
    strategy = ColumnTypeStrategy()
    with pytest.raises(TypeStrategyError):
        strategy.prepare_record({"unknown": 1})


def test_absent_key_in_record_is_treated_as_null(db):
    strategy = ColumnTypeStrategy()
    table = db["m"]
    strategy.apply_to_table(table, [{"a": 1, "b": 2}])
    strategy.apply_to_table(table, [{"a": 3}])  # b is absent, not NULL
    assert [dict(r) for r in table.rows] == [
        {"a": 1, "b": 2},
        {"a": 3, "b": None},
    ]


def test_apply_to_table_with_empty_batch_is_a_noop(db):
    strategy = ColumnTypeStrategy()
    table = db["m"]
    assert strategy.apply_to_table(table, []) is table
    assert not table.exists()
    assert strategy.batch_count == 0


def test_fixed_type_accepts_sql_string_and_python_type():
    strategy = ColumnTypeStrategy(fixed_types={"a": "INTEGER", "b": float})
    strategy.infer_batch([{"a": 1, "b": 2}])
    assert strategy.kind_for("a") is ColumnKind.INTEGER
    assert strategy.kind_for("b") is ColumnKind.REAL


def test_unknown_sql_type_raises():
    with pytest.raises(TypeStrategyError):
        ColumnTypeStrategy(fixed_types={"a": "WIBBLE"})


def test_demo_batches_show_state_storage_class_and_readback(db):
    """Short batches demonstrating the full lifecycle.

    Batch 1 looks all-integer; batch 2 introduces a decimal; batch 3 adds a
    leading-zero identifier and a big integer. At each step the inferred
    state, the real SQLite storage class and the read-back values are
    observable, and every change is explained in the decision log.
    """
    strategy = ColumnTypeStrategy(identifiers=["code"])
    table = db["measurements"]

    # Batch 1: integers only
    strategy.apply_to_table(table, [{"code": "0012", "temp": 21, "count": 1}])
    assert strategy.sql_types() == {
        "code": "TEXT",
        "temp": "INTEGER",
        "count": "INTEGER",
    }
    assert storage_classes(db, '"measurements"', "code", "temp", "count") == [
        ("text", "integer", "integer")
    ]

    # Batch 2: a decimal measurement widens temp to REAL
    strategy.apply_to_table(table, [{"code": "0007", "temp": 21.5, "count": 2}])
    assert strategy.sql_types()["temp"] == "REAL"
    assert [d.action for d in strategy.explain("temp")] == ["infer", "promote"]

    # Batch 3: a big integer cannot fit a 64-bit INTEGER, stored as exact text
    strategy.apply_to_table(
        table, [{"code": "12", "temp": 22.0, "count": 10**30}]
    )
    assert strategy.sql_types()["count"] == "TEXT"

    # Real SQLite storage classes and read-back values
    rows = db.execute(
        'SELECT code, typeof(code), temp, typeof(temp), count, typeof(count) '
        'FROM "measurements" ORDER BY rowid'
    ).fetchall()
    assert rows == [
        ("0012", "text", 21.0, "real", "1", "text"),
        ("0007", "text", 21.5, "real", "2", "text"),
        ("12", "text", 22.0, "real", "1000000000000000000000000000000", "text"),
    ]
    readback = [dict(r) for r in table.rows]
    assert readback[0]["code"] == "0012"
    assert readback[2]["count"] == "1000000000000000000000000000000"

    # The recorded reasons replay to the same schema
    replayed = ColumnTypeStrategy.from_log(strategy.to_log(), identifiers=["code"])
    assert replayed.sql_types() == strategy.sql_types()
    reasons = {d.column: d.reason for d in strategy.decisions if d.action == "promote"}
    assert "truncate" in reasons["temp"]


def test_legacy_text_column_can_promote_to_integer_when_clean(db):
    table = db["m"]
    table.insert({"x": "42"})  # pre-existing TEXT schema from an old import
    strategy = ColumnTypeStrategy()
    strategy.apply_to_table(table, [{"x": 7}])
    assert strategy.kind_for("x") is ColumnKind.INTEGER
    assert [c.type for c in table.columns] == ["INTEGER"]
    assert [r["x"] for r in table.rows] == [42, 7]


def test_case_insensitive_column_reuse(db):
    strategy = ColumnTypeStrategy()
    table = db["m"]
    strategy.apply_to_table(table, [{"Temp": 21}])
    strategy.apply_to_table(table, [{"temp": 21.5}])
    assert [c.name for c in table.columns] == ["Temp"]
    assert strategy.kind_for("temp") is ColumnKind.REAL
    assert [c.type for c in table.columns] == ["REAL"]


def test_last_batch_kinds_reports_per_batch_inference(db):
    strategy = ColumnTypeStrategy()
    strategy.apply_to_table(db["m"], [{"a": 1, "b": None}])
    assert strategy.last_batch_kinds == {"a": "INTEGER", "b": None}
    strategy.apply_to_table(db["m"], [{"a": 1.5}])
    assert strategy.last_batch_kinds["a"] == "REAL"


def test_fixed_text_serializes_decimal_and_datetime(db):
    strategy = ColumnTypeStrategy(fixed_types={"v": "TEXT"})
    table = db["m"]
    strategy.apply_to_table(
        table,
        [{"v": decimal.Decimal("1.25")}, {"v": datetime.date(2026, 1, 2)}],
    )
    assert [r["v"] for r in table.rows] == ["1.25", "2026-01-02"]
    assert storage_classes(db, "m", "v") == [("text",), ("text",)]


def test_any_fixed_column_keeps_native_values(db):
    strategy = ColumnTypeStrategy(fixed_types={"v": "ANY"})
    table = db["m"]
    strategy.apply_to_table(table, [{"v": b"\x01"}, {"v": 5}])
    assert strategy.kind_for("v") is ColumnKind.ANY
    assert storage_classes(db, "m", "v") == [("blob",), ("integer",)]


def test_identifier_int_values_are_stored_as_canonical_text(db):
    strategy = ColumnTypeStrategy(identifiers=["code"])
    table = db["m"]
    strategy.apply_to_table(table, [{"code": 12}, {"code": 1.5}])
    assert [r["code"] for r in table.rows] == ["12", "1.5"]
    assert storage_classes(db, "m", "code") == [("text",), ("text",)]


def test_public_symbols_are_exported():
    import sqlite_utils

    for symbol in (
        "ColumnTypeStrategy",
        "ColumnKind",
        "ColumnState",
        "Semantic",
        "TypeDecision",
        "TypeViolation",
        "TypeStrategyError",
        "BatchRejected",
    ):
        assert hasattr(sqlite_utils, symbol)
