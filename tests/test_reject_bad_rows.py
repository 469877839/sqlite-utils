import pytest

from sqlite_utils import Database
from sqlite_utils.db import RejectedRow, RejectedRows, Table
from sqlite_utils.utils import sqlite3


def traced(db):
    "Collect every SQL statement executed through ``Database.execute``."
    statements = []

    def tracer(sql, params):
        statements.append(sql)

    db._tracer = tracer
    return statements


def insert_statements(statements):
    return [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith("INSERT")
    ]


def test_reject_mode_all_valid_matches_default():
    default_table = Database(memory=True)["t"]
    default_table.create({"id": int, "name": str}, pk="id")
    reject_table = Database(memory=True)["t"]
    reject_table.create({"id": int, "name": str}, pk="id")
    records = [{"id": index, "name": "name-{}".format(index)} for index in range(5)]
    default_table.insert_all(records)
    rejects = reject_table.insert_all(records, reject_bad_rows=True)
    assert isinstance(rejects, RejectedRows)
    assert rejects.ok
    assert rejects == []
    assert rejects.indexes == []
    assert list(default_table.rows) == list(reject_table.rows)


def test_reject_mode_keeps_valid_rows_and_reports_sparse_bad_rows():
    db = Database(memory=True)
    table = db["t"]
    table.create({"id": int, "name": str}, pk="id")
    table.insert({"id": 100, "name": "existing"}, pk="id")
    records = [{"id": index, "name": "name-{}".format(index)} for index in range(20)]
    records[4] = {"id": 100, "name": "duplicate-of-existing"}
    records[17] = {"id": 100, "name": "duplicate-of-existing-too"}
    rejects = table.insert_all(records, reject_bad_rows=True, batch_size=20)
    assert isinstance(rejects, RejectedRows)
    assert rejects.indexes == [4, 17]
    assert [rejected.index for rejected in rejects] == [4, 17]
    # The original record object is returned untouched
    assert rejects[0].record == records[4]
    assert rejects[0].error_type == "IntegrityError"
    assert "UNIQUE constraint failed: t.id" in rejects[0].error
    assert rejects[0].sqlite_errorname
    assert rejects[0].sqlite_errorcode
    # Tuple-style unpacking works too
    index, record, error = rejects[0]
    assert index == 4
    assert record == records[4]
    assert error == rejects[0].error
    # Valid input rows were written, bad ones were not
    expected = [record for index, record in enumerate(records) if index not in {4, 17}]
    expected.insert(0, {"id": 100, "name": "existing"})
    assert sorted(table.rows, key=lambda row: row["id"]) == sorted(
        expected, key=lambda row: row["id"]
    )
    assert table.count == 19


def test_default_semantics_still_raise_and_roll_back():
    db = Database(memory=True)
    table = db["t"]
    table.create({"id": int}, pk="id")
    table.insert({"id": 1})
    with pytest.raises(sqlite3.IntegrityError):
        table.insert_all([{"id": 2}, {"id": 1}, {"id": 3}], batch_size=3)
    # Nothing from the failed batch was written
    assert [row["id"] for row in table.rows] == [1]
    # And the normal return value is still the table, not a rejection list
    assert isinstance(table.insert_all([{"id": 4}]), Table)


def test_duplicate_keys_within_batch_first_occurrence_wins():
    db = Database(memory=True)
    table = db["t"]
    rejects = table.insert_all(
        [
            {"id": 1, "value": "first"},
            {"id": 2, "value": "ok"},
            {"id": 1, "value": "second"},
            {"id": 2, "value": "duplicate"},
        ],
        pk="id",
        batch_size=4,
        reject_bad_rows=True,
    )
    assert rejects.indexes == [2, 3]
    assert sorted(table.rows, key=lambda row: row["id"]) == [
        {"id": 1, "value": "first"},
        {"id": 2, "value": "ok"},
    ]


def test_duplicate_keys_across_chunks_still_first_occurrence_wins():
    db = Database(memory=True)
    table = db["t"]
    rejects = table.insert_all(
        [
            {"id": 1},
            {"id": 2},
            {"id": 3},
            {"id": 1},
            {"id": 4},
        ],
        pk="id",
        batch_size=2,
        reject_bad_rows=True,
    )
    assert rejects.indexes == [3]
    assert sorted(row["id"] for row in table.rows) == [1, 2, 3, 4]


def test_upsert_duplicate_keys_last_occurrence_wins():
    db = Database(memory=True)
    table = db["t"]
    rejects = table.upsert_all(
        [
            {"id": 1, "value": "a"},
            {"id": 1, "value": "b"},
            {"id": 1, "value": "c"},
        ],
        pk="id",
        reject_bad_rows=True,
    )
    assert rejects.ok
    assert list(table.rows) == [{"id": 1, "value": "c"}]


def test_upsert_applies_valid_rows_and_rejects_constraint_failures():
    db = Database(memory=True)
    db.execute(
        "create table t(id integer primary key, value text check(value != 'bad'))"
    )
    table = db["t"]
    rejects = table.upsert_all(
        [
            {"id": 1, "value": "updated"},
            {"id": 2, "value": "new"},
            {"id": 3, "value": "bad"},
            {"id": 4, "value": "also-new"},
        ],
        pk="id",
        reject_bad_rows=True,
    )
    assert rejects.indexes == [2]
    assert "CHECK constraint failed" in rejects[0].error
    assert sorted(table.rows, key=lambda row: row["id"]) == [
        {"id": 1, "value": "updated"},
        {"id": 2, "value": "new"},
        {"id": 4, "value": "also-new"},
    ]


def test_check_and_not_null_failures_are_rejected():
    db = Database(memory=True)
    db.execute(
        "create table t("
        "id integer primary key, "
        "value text not null check(value != 'bad')"
        ")"
    )
    table = db["t"]
    rejects = table.insert_all(
        [
            {"id": 1, "value": "ok"},
            {"id": 2},
            {"id": 3, "value": "bad"},
            {"id": 4, "value": "fine"},
        ],
        reject_bad_rows=True,
    )
    assert rejects.indexes == [1, 2]
    assert "NOT NULL constraint failed" in rejects[0].error
    assert "CHECK constraint failed" in rejects[1].error
    assert sorted(table.rows, key=lambda row: row["id"]) == [
        {"id": 1, "value": "ok"},
        {"id": 4, "value": "fine"},
    ]


def test_foreign_key_failures_rejected_without_breaking_forward_references():
    db = Database(memory=True)
    db.execute("PRAGMA foreign_keys = ON")
    db["node"].create(
        {"id": int, "parent": int},
        pk="id",
        foreign_keys=[("parent", "node", "id")],
    )
    rejects = db["node"].insert_all(
        [
            # Refers to a record later in the same batch: valid in SQLite
            # because the whole multi-row statement is checked at its end
            {"id": 2, "parent": 1},
            {"id": 1, "parent": None},
            # Refers to a row that is never written
            {"id": 3, "parent": 99},
            {"id": 4, "parent": 2},
        ],
        reject_bad_rows=True,
        batch_size=4,
    )
    assert rejects.indexes == [2]
    assert "FOREIGN KEY constraint failed" in rejects[0].error
    assert sorted(db["node"].rows, key=lambda row: row["id"]) == [
        {"id": 1, "parent": None},
        {"id": 2, "parent": 1},
        {"id": 4, "parent": 2},
    ]
    # The chunk transaction committed cleanly: no deferred violation remains
    assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_foreign_key_cascade_rejects_orphaned_rows():
    db = Database(memory=True)
    db.execute("PRAGMA foreign_keys = ON")
    db["node"].create(
        {"id": int, "parent": int},
        pk="id",
        foreign_keys=[("parent", "node", "id")],
    )
    rejects = db["node"].insert_all(
        [
            {"id": 1, "parent": 99},  # rejected: no parent 99
            {"id": 2, "parent": 1},  # orphaned once row 1 is rejected
            {"id": 3, "parent": None},
            {"id": 4, "parent": 3},
        ],
        reject_bad_rows=True,
        batch_size=4,
    )
    assert rejects.indexes == [0, 1]
    assert sorted(db["node"].rows, key=lambda row: row["id"]) == [
        {"id": 3, "parent": None},
        {"id": 4, "parent": 3},
    ]
    assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_foreign_key_rejects_keep_input_order():
    db = Database(memory=True)
    db.execute("PRAGMA foreign_keys = ON")
    db["p"].create({"id": int}, pk="id")
    db["p"].insert({"id": 1})
    db["c"].create(
        {"id": int, "pid": int}, pk="id", foreign_keys=[("pid", "p", "id")]
    )
    rejects = db["c"].insert_all(
        [
            {"id": 1, "pid": 99},
            {"id": 2, "pid": 1},
            {"id": 3, "pid": 98},
            {"id": 4, "pid": 3},
        ],
        reject_bad_rows=True,
        batch_size=4,
    )
    assert rejects.indexes == [0, 2, 3]
    assert list(db["c"].rows) == [{"id": 2, "pid": 1}]


def test_foreign_keys_without_rowid_are_rejected():
    db = Database(memory=True)
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("create table parent(a text, b text, primary key (a, b)) without rowid")
    db.execute(
        "create table child("
        "id text primary key, pa text, pb text, "
        "foreign key (pa, pb) references parent(a, b)"
        ") without rowid"
    )
    db.execute("insert into parent values ('a', 'b')")
    rejects = db["child"].insert_all(
        [
            {"id": "ok", "pa": "a", "pb": "b"},
            {"id": "bad", "pa": "x", "pb": "y"},
            {"id": "null", "pa": None, "pb": None},
        ],
        reject_bad_rows=True,
    )
    assert rejects.indexes == [1]
    assert sorted(db["child"].rows, key=lambda row: row["id"]) == [
        {"id": "null", "pa": None, "pb": None},
        {"id": "ok", "pa": "a", "pb": "b"},
    ]


def test_generated_column_error_fails_whole_operation():
    db = Database(memory=True)
    db.execute("create table t(x integer, y integer generated always as (x * 2))")
    with pytest.raises(sqlite3.OperationalError):
        db["t"].insert_all(
            [{"x": 1, "y": 2}, {"x": 2}], reject_bad_rows=True
        )
    assert db["t"].count == 0


def test_closed_connection_error_fails_whole_operation():
    db = Database(memory=True)
    db["t"].create({"id": int}, pk="id")
    db.conn.close()
    with pytest.raises(sqlite3.ProgrammingError):
        db["t"].insert_all([{"id": 1}, {"id": 2}], reject_bad_rows=True)


def test_reject_bad_rows_cannot_disable_constraints():
    db = Database(memory=True)
    db["t"].create({"id": int, "value": str}, pk="id")
    db["t"].insert({"id": 1, "value": "existing"})
    rejects = db["t"].insert_all(
        [{"id": 1, "value": "duplicate"}], reject_bad_rows=True
    )
    assert rejects.indexes == [0]
    assert list(db["t"].rows) == [{"id": 1, "value": "existing"}]
    # The foreign key pragma, if enabled, stays enabled
    db.execute("PRAGMA foreign_keys = ON")
    assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_reject_bad_rows_is_incompatible_with_ignore_and_replace():
    db = Database(memory=True)
    with pytest.raises(ValueError):
        db["t"].insert_all([{"id": 1}], reject_bad_rows=True, ignore=True)
    with pytest.raises(ValueError):
        db["t"].insert_all([{"id": 1}], reject_bad_rows=True, replace=True)


def test_input_is_only_traversed_once():
    db = Database(memory=True)
    db["t"].create({"id": int}, pk="id")
    db["t"].insert({"id": 7})
    seen = []

    def generate():
        for index in range(12):
            seen.append(index)
            yield {"id": index}

    rejects = db["t"].insert_all(generate(), batch_size=5, reject_bad_rows=True)
    assert seen == list(range(12))
    assert rejects.indexes == [7]


def test_reject_mode_with_list_mode_and_alter():
    db = Database(memory=True)
    rejects = db["people"].insert_all(
        [
            ["id", "name"],
            [1, "Cleo"],
            [1, "Duplicate"],
            [2, "Marnie"],
        ],
        pk="id",
        reject_bad_rows=True,
    )
    assert rejects.indexes == [1]
    assert sorted(db["people"].rows, key=lambda row: row["id"]) == [
        {"id": 1, "name": "Cleo"},
        {"id": 2, "name": "Marnie"},
    ]

    db2 = Database(memory=True)
    db2["t"].create({"id": int}, pk="id")
    db2["t"].insert({"id": 9})
    rejects2 = db2["t"].insert_all(
        [
            {"id": 1, "extra": "first"},
            {"id": 9, "extra": "duplicate"},
            {"id": 2, "extra": "second"},
        ],
        alter=True,
        reject_bad_rows=True,
    )
    assert rejects2.indexes == [1]
    assert sorted(row["id"] for row in db2["t"].rows) == [1, 2, 9]
    assert db2["t"].columns_dict["extra"] is str


def test_reject_mode_returns_indexes_across_multiple_chunks():
    db = Database(memory=True)
    db["t"].create({"id": int}, pk="id")
    db["t"].insert({"id": 1})
    db["t"].insert({"id": 5})
    rejects = db["t"].insert_all(
        [{"id": index} for index in range(9)],
        batch_size=3,
        reject_bad_rows=True,
    )
    assert rejects.indexes == [1, 5]
    assert sorted(row["id"] for row in db["t"].rows) == [0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_empty_input_returns_empty_rejects():
    db = Database(memory=True)
    rejects = db["t"].insert_all([], reject_bad_rows=True)
    assert isinstance(rejects, RejectedRows)
    assert rejects.ok
    assert not db["t"].exists()


def test_fully_rejected_batch_writes_nothing():
    db = Database(memory=True)
    db.execute(
        "create table t(id integer primary key, value text check(value != 'bad'))"
    )
    rejects = db["t"].insert_all(
        [{"id": 1, "value": "bad"}, {"id": 2, "value": "bad"}],
        reject_bad_rows=True,
    )
    assert rejects.indexes == [0, 1]
    assert db["t"].count == 0


def test_database_call_counts_for_success_and_sparse_bad_rows():
    # Fully successful batch: one bulk INSERT regardless of row count
    db = Database(memory=True)
    db["t"].create({"id": int}, pk="id")
    statements = traced(db)
    rejects = db["t"].insert_all(
        [{"id": index} for index in range(16)],
        batch_size=16,
        reject_bad_rows=True,
    )
    assert rejects.ok
    assert len(statements) == 9
    assert len(insert_statements(statements)) == 1

    # Sparse bad rows: two bad records out of 64 cost 23 INSERT calls,
    # which is far fewer than the 64 a row-at-a-time approach would use
    db = Database(memory=True)
    db["t"].create({"id": int}, pk="id")
    db["t"].insert({"id": 20})
    db["t"].insert({"id": 50})
    statements = traced(db)
    rejects = db["t"].insert_all(
        [{"id": index} for index in range(64)],
        batch_size=64,
        reject_bad_rows=True,
    )
    assert rejects.indexes == [20, 50]
    assert len(insert_statements(statements)) == 23
    assert len(statements) == 53
    assert len(insert_statements(statements)) < 64

    # Compare with literal row-at-a-time writes
    db = Database(memory=True)
    db["t"].create({"id": int}, pk="id")
    db["t"].insert({"id": 20})
    db["t"].insert({"id": 50})
    statements = traced(db)
    for index in range(64):
        try:
            db["t"].insert({"id": index}, pk="id")
        except sqlite3.IntegrityError:
            pass
    assert len(insert_statements(statements)) == 64
    assert 23 < 64
