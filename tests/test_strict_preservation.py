"""Tests for strict schema preservation in table.transform(preserve="strict").

These tests build small databases, transform them and then compare the full
sqlite_schema contents and the actual behaviour of indexes, triggers and
views - a row count alone cannot prove a rebuild preserved the schema.
"""

import re

import pytest

from sqlite_utils.db import Database
from sqlite_utils.schema_preservation import PreservationError


@pytest.fixture
def fresh_db():
    return Database(memory=True)


def schema_objects(db):
    return {
        (row[0], row[1]): row[2]
        for row in db.execute(
            "SELECT type, name, sql FROM sqlite_schema ORDER BY type, name"
        ).fetchall()
    }


def compact(sql):
    return re.sub(r"\s+", " ", sql).strip()


def make_database(db):
    db.executescript("""
        CREATE TABLE t (
            id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            "full name" TEXT
        );
        CREATE TABLE log (msg TEXT);
        INSERT INTO t VALUES (1, 'cleo', 5, 'Cleo Pancakes');
        CREATE INDEX idx_name ON t(name);
        CREATE INDEX idx_expr ON t(lower(name));
        CREATE INDEX idx_partial ON t(age) WHERE name IS NOT NULL;
        CREATE INDEX idx_quoted ON t("full name");
        CREATE TRIGGER trg AFTER INSERT ON t
        BEGIN
            INSERT INTO log VALUES (NEW.name);
        END;
        CREATE VIEW v_names AS SELECT id, name FROM t;
        CREATE VIEW v_unrelated AS SELECT msg FROM log;
    """)
    return db


def test_strict_rename_preserves_schema_and_behaviour(fresh_db):
    db = make_database(fresh_db)
    db["t"].transform(rename={"name": "full_name"}, preserve="strict")

    assert db["t"].columns_dict == {
        "id": int,
        "full_name": str,
        "age": int,
        "full name": str,
    }
    objects = schema_objects(db)
    assert compact(objects[("index", "idx_name")]) == (
        "CREATE INDEX idx_name ON t(full_name)"
    )
    assert compact(objects[("index", "idx_expr")]) == (
        "CREATE INDEX idx_expr ON t(lower(full_name))"
    )
    assert compact(objects[("index", "idx_partial")]) == (
        "CREATE INDEX idx_partial ON t(age) WHERE full_name IS NOT NULL"
    )
    assert compact(objects[("index", "idx_quoted")]) == (
        'CREATE INDEX idx_quoted ON t("full name")'
    )
    assert compact(objects[("trigger", "trg")]) == (
        "CREATE TRIGGER trg AFTER INSERT ON t BEGIN "
        "INSERT INTO log VALUES (NEW.full_name); END"
    )
    assert compact(objects[("view", "v_names")]) == (
        "CREATE VIEW v_names AS SELECT id, full_name FROM t"
    )
    # Unrelated view is byte-for-byte untouched
    assert objects[("view", "v_unrelated")] == (
        "CREATE VIEW v_unrelated AS SELECT msg FROM log"
    )

    # Trigger still fires against the renamed column
    db.execute("INSERT INTO t VALUES (2, 'rex', 3, 'Rex')")
    assert list(db["log"].rows) == [{"msg": "rex"}]
    # Expression index is used and correct
    assert list(db.query("SELECT id FROM t WHERE lower(full_name) = 'rex'")) == [
        {"id": 2}
    ]
    # Partial index still enforces its predicate
    assert sorted(
        row["age"]
        for row in db.query("SELECT age FROM t WHERE full_name IS NOT NULL")
    ) == [3, 5]
    # View reads the renamed column
    assert list(db.query("SELECT * FROM v_names")) == [
        {"id": 1, "full_name": "cleo"},
        {"id": 2, "full_name": "rex"},
    ]


def test_strict_rename_uses_expression_index(fresh_db):
    db = make_database(fresh_db)
    db["t"].transform(rename={"name": "full_name"}, preserve="strict")
    plan = db.execute(
        "EXPLAIN QUERY PLAN SELECT id FROM t WHERE lower(full_name) = 'cleo'"
    ).fetchall()
    assert any("idx_expr" in row[3] for row in plan)


def test_strict_drop_column_with_safe_objects(fresh_db):
    db = Database(memory=True)
    db.executescript("""
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, note TEXT);
        CREATE TABLE log (msg TEXT);
        INSERT INTO t VALUES (1, 'cleo', 'first');
        CREATE INDEX idx_name ON t(name);
        CREATE TRIGGER trg AFTER INSERT ON t
        BEGIN
            INSERT INTO log VALUES (NEW.name);
        END;
        CREATE VIEW v_names AS SELECT id, name FROM t;
    """)
    db["t"].transform(drop=["note"], preserve="strict")
    assert db["t"].columns_dict == {"id": int, "name": str}
    objects = schema_objects(db)
    assert ("table", "t") in objects
    assert compact(objects[("index", "idx_name")]) == "CREATE INDEX idx_name ON t(name)"
    assert compact(objects[("trigger", "trg")]) == (
        "CREATE TRIGGER trg AFTER INSERT ON t BEGIN "
        "INSERT INTO log VALUES (NEW.name); END"
    )
    assert compact(objects[("view", "v_names")]) == (
        "CREATE VIEW v_names AS SELECT id, name FROM t"
    )
    db.execute("INSERT INTO t VALUES (2, 'rex')")
    assert list(db["log"].rows) == [{"msg": "rex"}]


def test_strict_refuses_drop_of_indexed_column(fresh_db):
    db = make_database(fresh_db)
    before = schema_objects(db)
    with pytest.raises(PreservationError, match="index 'idx_partial'"):
        db["t"].transform(drop=["age"], preserve="strict")
    assert schema_objects(db) == before
    assert db["t"].columns_dict == {
        "id": int,
        "name": str,
        "age": int,
        "full name": str,
    }


def test_strict_refuses_drop_of_expression_index_column(fresh_db):
    db = Database(memory=True)
    db.executescript("""
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT);
        INSERT INTO t VALUES (1, 'cleo');
        CREATE INDEX idx_expr ON t(lower(name));
    """)
    before = schema_objects(db)
    with pytest.raises(PreservationError, match="index 'idx_expr'"):
        db["t"].transform(drop=["name"], preserve="strict")
    assert schema_objects(db) == before


def test_strict_refuses_drop_of_trigger_column(fresh_db):
    db = Database(memory=True)
    db.executescript("""
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE log (msg TEXT);
        INSERT INTO t VALUES (1, 'cleo');
        CREATE TRIGGER trg AFTER INSERT ON t
        BEGIN
            INSERT INTO log VALUES (NEW.name);
        END;
    """)
    before = schema_objects(db)
    with pytest.raises(PreservationError, match="trigger 'trg'"):
        db["t"].transform(drop=["name"], preserve="strict")
    assert schema_objects(db) == before
    # Trigger still works on the untouched database
    db.execute("INSERT INTO t VALUES (2, 'rex')")
    assert list(db["log"].rows) == [{"msg": "rex"}]


def test_strict_refuses_drop_of_view_column(fresh_db):
    db = Database(memory=True)
    db.executescript("""
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT);
        INSERT INTO t VALUES (1, 'cleo');
        CREATE VIEW v_names AS SELECT id, name FROM t;
    """)
    before = schema_objects(db)
    with pytest.raises(PreservationError, match="view 'v_names'"):
        db["t"].transform(drop=["name"], preserve="strict")
    assert schema_objects(db) == before
    assert list(db.query("SELECT * FROM v_names")) == [{"id": 1, "name": "cleo"}]


def test_strict_refuses_ambiguous_join_view(fresh_db):
    db = make_database(fresh_db)
    db.execute("CREATE TABLE other (id INTEGER, name TEXT)")
    db.execute(
        "CREATE VIEW v_join AS SELECT name FROM t JOIN other ON other.id = t.id"
    )
    before = schema_objects(db)
    with pytest.raises(PreservationError, match="view 'v_join'"):
        db["t"].transform(rename={"name": "full_name"}, preserve="strict")
    assert schema_objects(db) == before


def test_strict_refuses_trigger_subquery_reference(fresh_db):
    db = Database(memory=True)
    db.executescript("""
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        INSERT INTO t VALUES (1, 'cleo', 5);
        CREATE TRIGGER trg2 AFTER INSERT ON t BEGIN
            UPDATE t SET age = (SELECT max(age) FROM t WHERE name IS NOT NULL);
        END;
    """)
    before = schema_objects(db)
    with pytest.raises(PreservationError, match="trigger 'trg2'"):
        db["t"].transform(rename={"name": "full_name"}, preserve="strict")
    assert schema_objects(db) == before


def test_strict_lenient_default_unchanged(fresh_db):
    # The lenient default keeps its historical behaviour - strict is opt-in.
    db = Database(memory=True)
    db.executescript("""
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT);
        INSERT INTO t VALUES (1, 'cleo');
        CREATE INDEX idx_name ON t(name);
        CREATE TRIGGER trg AFTER INSERT ON t BEGIN SELECT 1; END;
        CREATE VIEW v_names AS SELECT id, name FROM t;
    """)
    db["t"].transform(rename={"name": "full_name"})
    objects = schema_objects(db)
    assert ("trigger", "trg") not in objects
    assert compact(objects[("index", "idx_name")]) == (
        'CREATE INDEX "idx_name" ON "t" ("full_name")'
    )
    # The lenient view keeps pointing at the old column name
    assert compact(objects[("view", "v_names")]) == (
        "CREATE VIEW v_names AS SELECT id, name FROM t"
    )


def test_strict_invalid_preserve_value(fresh_db):
    db = make_database(fresh_db)
    with pytest.raises(ValueError, match="preserve"):
        db["t"].transform(rename={"name": "x"}, preserve="bogus")


def test_strict_rollback_on_failure(fresh_db):
    # A failure partway through the rebuild must leave the database untouched.
    db = Database(memory=True)
    db.executescript("""
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        CREATE INDEX idx_name ON t(name);
        INSERT INTO t VALUES (1, 'cleo', NULL);
    """)
    before = schema_objects(db)
    with pytest.raises(Exception):
        db["t"].transform(not_null={"age": True}, preserve="strict")
    assert schema_objects(db) == before
    assert db["t"].count == 1


def test_strict_trigger_update_of_and_when(fresh_db):
    db = Database(memory=True)
    db.executescript("""
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        CREATE TABLE log (msg TEXT);
        INSERT INTO t VALUES (1, 'cleo', 5);
        CREATE TRIGGER trg AFTER UPDATE OF name ON t
        WHEN NEW.name IS NOT NULL
        BEGIN
            INSERT INTO log VALUES (NEW.name);
        END;
    """)
    db["t"].transform(rename={"name": "full_name"}, preserve="strict")
    objects = schema_objects(db)
    assert compact(objects[("trigger", "trg")]) == (
        "CREATE TRIGGER trg AFTER UPDATE OF full_name ON t "
        "WHEN NEW.full_name IS NOT NULL BEGIN "
        "INSERT INTO log VALUES (NEW.full_name); END"
    )
    # Trigger only fires on updates of the renamed column
    db.execute("UPDATE t SET age = 6 WHERE id = 1")
    assert list(db["log"].rows) == []
    db.execute("UPDATE t SET full_name = 'rex' WHERE id = 1")
    assert list(db["log"].rows) == [{"msg": "rex"}]


def test_strict_trigger_updating_transformed_table(fresh_db):
    db = Database(memory=True)
    db.executescript("""
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, upper_name TEXT);
        INSERT INTO t VALUES (1, 'cleo', NULL);
        CREATE TRIGGER trg AFTER INSERT ON t
        BEGIN
            UPDATE t SET upper_name = upper(NEW.name) WHERE id = NEW.id;
        END;
    """)
    db["t"].transform(rename={"name": "full_name"}, preserve="strict")
    objects = schema_objects(db)
    assert compact(objects[("trigger", "trg")]) == (
        "CREATE TRIGGER trg AFTER INSERT ON t BEGIN "
        "UPDATE t SET upper_name = upper(NEW.full_name) WHERE id = NEW.id; END"
    )
    db.execute("INSERT INTO t VALUES (2, 'rex', NULL)")
    assert list(db.query("SELECT upper_name FROM t WHERE id = 2")) == [
        {"upper_name": "REX"}
    ]


def test_strict_keep_table(fresh_db):
    db = make_database(fresh_db)
    db["t"].transform(
        rename={"name": "full_name"}, keep_table="t_backup", preserve="strict"
    )
    assert db["t"].columns_dict == {
        "id": int,
        "full_name": str,
        "age": int,
        "full name": str,
    }
    assert "t_backup" in db.table_names()
    objects = schema_objects(db)
    # Trigger was recreated on the new table, not left on the backup
    assert compact(objects[("trigger", "trg")]) == (
        "CREATE TRIGGER trg AFTER INSERT ON t BEGIN "
        "INSERT INTO log VALUES (NEW.full_name); END"
    )
    db.execute("INSERT INTO t VALUES (2, 'rex', 3, 'Rex')")
    assert list(db["log"].rows) == [{"msg": "rex"}]


def test_strict_transform_sql_ordering(fresh_db):
    db = make_database(fresh_db)
    sqls = db["t"].transform_sql(rename={"name": "full_name"}, preserve="strict")
    drop_table = sqls.index('DROP TABLE "t";')
    for statement in sqls:
        if statement.startswith(("DROP INDEX", "DROP TRIGGER", "DROP VIEW")):
            assert sqls.index(statement) < drop_table
    create_trigger = next(s for s in sqls if s.startswith("CREATE TRIGGER"))
    create_view = next(s for s in sqls if s.startswith("CREATE VIEW"))
    rename_to = next(s for s in sqls if "RENAME TO" in s and "t_new" in s)
    assert sqls.index(rename_to) < sqls.index(create_trigger)
    assert sqls.index(rename_to) < sqls.index(create_view)


def test_strict_composite_unique_constraint(fresh_db):
    db = Database(memory=True)
    db.executescript("""
        CREATE TABLE memberships (
            account_id INTEGER,
            email TEXT,
            CONSTRAINT unique_membership
                UNIQUE (account_id DESC, email COLLATE NOCASE)
                ON CONFLICT ABORT
        );
        INSERT INTO memberships VALUES (1, 'a@example.com');
        CREATE INDEX idx_email ON memberships(email);
    """)
    db["memberships"].transform(rename={"email": "address"}, preserve="strict")
    schema = db["memberships"].schema
    assert "UNIQUE" in schema and "address" in schema
    objects = schema_objects(db)
    assert compact(objects[("index", "idx_email")]) == (
        "CREATE INDEX idx_email ON memberships(address)"
    )
    # The composite constraint still rejects duplicates
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO memberships VALUES (1, 'A@EXAMPLE.COM')")


def test_strict_cli(tmp_path):
    from click.testing import CliRunner

    from sqlite_utils import cli

    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.executescript("""
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE log (msg TEXT);
        INSERT INTO t VALUES (1, 'cleo');
        CREATE TRIGGER trg AFTER INSERT ON t
        BEGIN
            INSERT INTO log VALUES (NEW.name);
        END;
    """)
    db.close()
    result = CliRunner().invoke(
        cli.cli,
        [
            "transform",
            db_path,
            "t",
            "--rename",
            "name",
            "full_name",
            "--preserve",
            "strict",
        ],
    )
    assert result.exit_code == 0, result.output
    db = Database(db_path)
    assert db["t"].columns_dict == {"id": int, "full_name": str}
    objects = schema_objects(db)
    assert compact(objects[("trigger", "trg")]) == (
        "CREATE TRIGGER trg AFTER INSERT ON t BEGIN "
        "INSERT INTO log VALUES (NEW.full_name); END"
    )
    db.close()


def test_strict_cli_refusal(tmp_path):
    from click.testing import CliRunner

    from sqlite_utils import cli

    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.executescript("""
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE log (msg TEXT);
        INSERT INTO t VALUES (1, 'cleo');
        CREATE TRIGGER trg AFTER INSERT ON t
        BEGIN
            INSERT INTO log VALUES (NEW.name);
        END;
    """)
    db.close()
    result = CliRunner().invoke(
        cli.cli,
        ["transform", db_path, "t", "--drop", "name", "--preserve", "strict"],
    )
    assert result.exit_code == 1
    assert "trigger 'trg'" in result.output
    db = Database(db_path)
    assert db["t"].columns_dict == {"id": int, "name": str}
    assert db["t"].count == 1
    db.close()


def test_strict_cross_table_trigger_rewritten(fresh_db):
    db = Database(memory=True)
    db.executescript("""
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE other (id INTEGER PRIMARY KEY);
        INSERT INTO t VALUES (1, 'cleo');
        CREATE TRIGGER trg_other AFTER INSERT ON other BEGIN
            INSERT INTO t(name) VALUES ('new');
        END;
        CREATE TRIGGER trg_other2 AFTER INSERT ON other BEGIN
            UPDATE t SET name = 'z' WHERE id = NEW.id;
        END;
    """)
    db["t"].transform(rename={"name": "full_name"}, preserve="strict")
    objects = schema_objects(db)
    assert compact(objects[("trigger", "trg_other")]) == (
        "CREATE TRIGGER trg_other AFTER INSERT ON other BEGIN "
        "INSERT INTO t(full_name) VALUES ('new'); END"
    )
    assert compact(objects[("trigger", "trg_other2")]) == (
        "CREATE TRIGGER trg_other2 AFTER INSERT ON other BEGIN "
        "UPDATE t SET full_name = 'z' WHERE id = NEW.id; END"
    )
    # Both triggers still fire correctly
    db.execute("INSERT INTO other VALUES (5)")
    assert list(db.query("SELECT full_name FROM t ORDER BY id")) == [
        {"full_name": "cleo"},
        {"full_name": "new"},
    ]


def test_strict_refuses_cross_table_trigger_drop(fresh_db):
    db = Database(memory=True)
    db.executescript("""
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE other (id INTEGER PRIMARY KEY);
        INSERT INTO t VALUES (1, 'cleo');
        CREATE TRIGGER trg_other AFTER INSERT ON other BEGIN
            INSERT INTO t(name) VALUES ('new');
        END;
    """)
    before = schema_objects(db)
    with pytest.raises(PreservationError, match="trigger 'trg_other'"):
        db["t"].transform(drop=["name"], preserve="strict")
    assert schema_objects(db) == before


def test_strict_unrelated_cross_table_trigger_untouched(fresh_db):
    db = Database(memory=True)
    db.executescript("""
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE other (id INTEGER PRIMARY KEY, note TEXT);
        CREATE TRIGGER trg_u AFTER INSERT ON other BEGIN
            UPDATE other SET note = 'fired' WHERE id = NEW.id;
        END;
    """)
    before = db.execute(
        "SELECT sql FROM sqlite_schema WHERE name='trg_u'"
    ).fetchone()[0]
    db["t"].transform(rename={"name": "full_name"}, preserve="strict")
    after = db.execute(
        "SELECT sql FROM sqlite_schema WHERE name='trg_u'"
    ).fetchone()[0]
    assert before == after


def test_strict_refuses_fts_sync_trigger(fresh_db):
    # FTS external-content triggers insert into a different table, so strict
    # mode refuses rather than guessing how to rewrite them.
    fresh_db["t"].insert_all(
        [{"id": 1, "name": "cleo", "age": 5}], pk="id"
    )
    fresh_db["t"].enable_fts(
        ["name"], fts_version="FTS4", create_triggers=True
    )
    before = schema_objects(fresh_db)
    with pytest.raises(PreservationError, match="trigger 't_ai'"):
        fresh_db["t"].transform(rename={"name": "full_name"}, preserve="strict")
    assert schema_objects(fresh_db) == before


def test_strict_refuses_instead_of_trigger_on_dependent_view(fresh_db):
    db = Database(memory=True)
    db.executescript("""
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT);
        CREATE VIEW v AS SELECT * FROM t;
        CREATE TRIGGER trv INSTEAD OF INSERT ON v BEGIN
            INSERT INTO t(name) VALUES (NEW.name);
        END;
        INSERT INTO t VALUES (1, 'cleo');
    """)
    before = schema_objects(db)
    with pytest.raises(PreservationError, match="trigger 'trv'"):
        db["t"].transform(rename={"name": "full_name"}, preserve="strict")
    assert schema_objects(db) == before


def test_strict_instead_of_trigger_survives_unrelated_transform(fresh_db):
    db = Database(memory=True)
    db.executescript("""
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT);
        CREATE VIEW v AS SELECT * FROM t;
        CREATE TRIGGER trv INSTEAD OF INSERT ON v BEGIN
            INSERT INTO t(name) VALUES (NEW.name);
        END;
    """)
    db["t"].transform(types={"id": str}, preserve="strict")
    assert compact(schema_objects(db)[("trigger", "trv")]) == (
        "CREATE TRIGGER trv INSTEAD OF INSERT ON v BEGIN "
        "INSERT INTO t(name) VALUES (NEW.name); END"
    )
