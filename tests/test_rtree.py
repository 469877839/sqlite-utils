# RTree spatial index tests - use only the SQLite rtree module, no SpatiaLite
import json
import math
import random
import struct

import pytest
from click.testing import CliRunner

import sqlite_utils
from sqlite_utils.cli import cli
from sqlite_utils.db import Database, InvalidColumns, InvalidRectangle, NoTable
from sqlite_utils.utils import sqlite3

RTREE_COLUMNS = ("minx", "maxx", "miny", "maxy")
F32_MIN_NORMAL = 1.1754943508222875e-38
F32_MAX = 3.4028234663852886e38


def make_table(db, name="boxes", **kwargs):
    table = db.table(name)
    table.create(
        {
            "id": int,
            "name": str,
            "minx": float,
            "maxx": float,
            "miny": float,
            "maxy": float,
        },
        pk="id",
        **kwargs,
    )
    return table


def rtree_ids(db, index_name="rtree_boxes"):
    return {row[0] for row in db.execute(f"SELECT id FROM {index_name}").fetchall()}


# Independent reference implementation: compare against the original table
# values in pure Python, never looking at the RTree index.
def reference_matches(rows, bounds, mode):
    minx, maxx, miny, maxy = bounds
    out = []
    for row in rows:
        values = [row[c] for c in RTREE_COLUMNS]
        if any(v is None for v in values):
            continue
        if any(math.isnan(v) for v in values if isinstance(v, float)):
            continue
        rminx, rmaxx, rminy, rmaxy = values
        if mode == "intersects":
            if (
                rminx <= maxx
                and rmaxx >= minx
                and rminy <= maxy
                and rmaxy >= miny
            ):
                out.append(row["id"])
        elif mode == "contains":
            if (
                rminx <= minx
                and rmaxx >= maxx
                and rminy <= miny
                and rmaxy >= maxy
            ):
                out.append(row["id"])
        else:
            raise AssertionError(mode)
    return set(out)


@pytest.fixture
def populated(fresh_db):
    table = make_table(fresh_db)
    table.enable_rtree(list(RTREE_COLUMNS))
    table.insert_all(
        [
            # ordinary box, zero-area point, negative box, NULL box, zero point
            {"id": 1, "name": "a", "minx": 0.0, "maxx": 10.0, "miny": 0.0, "maxy": 10.0},
            {"id": 2, "name": "point", "minx": 5.0, "maxx": 5.0, "miny": 5.0, "maxy": 5.0},
            {"id": 3, "name": "neg", "minx": -20.0, "maxx": -10.0, "miny": -20.0, "maxy": -10.0},
            {"id": 4, "name": "nullbox", "minx": None, "maxx": 1.0, "miny": 0.0, "maxy": 1.0},
            {"id": 5, "name": "zero", "minx": 0.0, "maxx": 0.0, "miny": 0.0, "maxy": 0.0},
        ]
    )
    return table


# --- creation and schema ----------------------------------------------------

def test_enable_rtree_creates_virtual_table_and_triggers(fresh_db):
    table = make_table(fresh_db)
    table.insert({"id": 1, "minx": 0.0, "maxx": 1.0, "miny": 0.0, "maxy": 1.0})
    table.enable_rtree(list(RTREE_COLUMNS))
    assert fresh_db["rtree_boxes"].virtual_table_using == "RTREE"
    assert table.detect_rtree() == ["rtree_boxes"]
    trigger_names = {t.name for t in table.triggers}
    assert trigger_names == {
        "rtree_boxes_bi",
        "rtree_boxes_bu",
        "rtree_boxes_ai",
        "rtree_boxes_au",
        "rtree_boxes_ad",
    }
    assert fresh_db["rtree_boxes_rowid"].exists()
    assert fresh_db["rtree_boxes_node"].exists()
    assert fresh_db["rtree_boxes_parent"].exists()
    meta = fresh_db["_rtree_columns"].get("rtree_boxes")
    assert json.loads(meta["columns"]) == list(RTREE_COLUMNS)
    assert meta["table_name"] == "boxes"


def test_enable_rtree_never_changes_business_columns(fresh_db):
    table = make_table(fresh_db)
    before = [(c.name, c.type, c.notnull) for c in table.columns]
    table.enable_rtree(list(RTREE_COLUMNS))
    after = [(c.name, c.type, c.notnull) for c in table.columns]
    assert before == after
    assert table.schema.startswith('CREATE TABLE "boxes"')


def test_enable_rtree_indexes_existing_rows_and_skips_nulls(fresh_db):
    table = make_table(fresh_db)
    table.insert_all(
        [
            {"id": 1, "minx": 0.0, "maxx": 1.0, "miny": 0.0, "maxy": 1.0},
            {"id": 2, "minx": None, "maxx": 1.0, "miny": 0.0, "maxy": 1.0},
        ]
    )
    table.enable_rtree(list(RTREE_COLUMNS))
    assert rtree_ids(fresh_db) == {1}


def test_enable_rtree_is_idempotent_and_replaceable(fresh_db):
    table = make_table(fresh_db)
    table.insert({"id": 1, "minx": 0.0, "maxx": 1.0, "miny": 0.0, "maxy": 1.0})
    table.enable_rtree(list(RTREE_COLUMNS))
    table.enable_rtree(list(RTREE_COLUMNS))
    assert table.detect_rtree() == ["rtree_boxes"]
    table.enable_rtree(list(RTREE_COLUMNS), replace=True)
    assert rtree_ids(fresh_db) == {1}
    assert len(table.triggers) == 5


def test_enable_rtree_custom_index_name(fresh_db):
    table = make_table(fresh_db)
    table.enable_rtree(list(RTREE_COLUMNS), index_name="spatial_idx")
    assert table.detect_rtree() == ["spatial_idx"]
    assert fresh_db["spatial_idx"].virtual_table_using == "RTREE"


def test_enable_rtree_requires_existing_table(fresh_db):
    with pytest.raises(NoTable):
        fresh_db["missing"].enable_rtree(list(RTREE_COLUMNS))


def test_enable_rtree_requires_four_columns(fresh_db):
    table = make_table(fresh_db)
    with pytest.raises(InvalidColumns):
        table.enable_rtree(["minx", "maxx", "miny"])
    with pytest.raises(InvalidColumns):
        table.enable_rtree(["minx", "maxx", "miny", "maxy", "minx"])


def test_enable_rtree_rejects_unknown_and_nonnumeric_columns(fresh_db):
    table = make_table(fresh_db)
    with pytest.raises(InvalidColumns):
        table.enable_rtree(["nope", "maxx", "miny", "maxy"])
    with pytest.raises(InvalidColumns):
        table.enable_rtree(["name", "maxx", "miny", "maxy"])


def test_enable_rtree_rejects_without_rowid(fresh_db):
    fresh_db.execute(
        "CREATE TABLE w (id INTEGER PRIMARY KEY, minx REAL, maxx REAL, "
        "miny REAL, maxy REAL) WITHOUT ROWID"
    )
    with pytest.raises(ValueError):
        fresh_db["w"].enable_rtree(list(RTREE_COLUMNS))


def test_enable_rtree_rejects_inverted_existing_data(fresh_db):
    table = make_table(fresh_db)
    table.insert({"id": 1, "minx": 9.0, "maxx": 1.0, "miny": 0.0, "maxy": 1.0})
    with pytest.raises(InvalidRectangle) as exc:
        table.enable_rtree(list(RTREE_COLUMNS))
    assert "inverted" in str(exc.value)
    assert "Row 1" in str(exc.value)
    # Failed creation rolls everything back
    assert not fresh_db["rtree_boxes"].exists()
    assert not fresh_db["_rtree_columns"].exists()
    assert table.triggers == []


def test_enable_rtree_rejects_out_of_range_existing_data(fresh_db):
    table = make_table(fresh_db)
    table.insert({"id": 1, "minx": 0.0, "maxx": 1e39, "miny": 0.0, "maxy": 1.0})
    with pytest.raises(InvalidRectangle):
        table.enable_rtree(list(RTREE_COLUMNS))
    assert not fresh_db["rtree_boxes"].exists()


def test_enable_rtree_works_on_plain_rowid_table(fresh_db):
    fresh_db.execute(
        "CREATE TABLE p (minx REAL, maxx REAL, miny REAL, maxy REAL)"
    )
    fresh_db["p"].enable_rtree(list(RTREE_COLUMNS))
    fresh_db.execute("INSERT INTO p VALUES (0, 1, 0, 1)")
    assert fresh_db.execute("SELECT id FROM rtree_p").fetchone()[0] == 1


# --- trigger synchronization ------------------------------------------------

def test_insert_syncs_index(populated):
    populated.insert(
        {"id": 6, "minx": 100.0, "maxx": 110.0, "miny": 100.0, "maxy": 110.0}
    )
    assert rtree_ids(populated.db) == {1, 2, 3, 5, 6}
    row = populated.db.execute(
        "SELECT minx, maxx FROM rtree_boxes WHERE id = 6"
    ).fetchone()
    assert row == (100.0, 110.0)


def test_insert_with_null_coordinates_is_not_indexed(populated):
    populated.insert(
        {"id": 6, "minx": None, "maxx": None, "miny": None, "maxy": None}
    )
    assert 6 not in rtree_ids(populated.db)


def test_update_syncs_index(populated):
    populated.update(2, {"minx": 50.0, "maxx": 60.0, "miny": 50.0, "maxy": 60.0})
    row = populated.db.execute(
        "SELECT minx, maxx FROM rtree_boxes WHERE id = 2"
    ).fetchone()
    assert row == (50.0, 60.0)


def test_update_to_null_removes_index_entry(populated):
    populated.update(2, {"minx": None})
    assert 2 not in rtree_ids(populated.db)
    # and back again
    populated.update(2, {"minx": 5.0})
    assert 2 in rtree_ids(populated.db)


def test_update_unrelated_column_leaves_index_untouched(populated):
    before = populated.db.execute(
        "SELECT * FROM rtree_boxes ORDER BY id"
    ).fetchall()
    populated.update(2, {"name": "renamed"})
    after = populated.db.execute(
        "SELECT * FROM rtree_boxes ORDER BY id"
    ).fetchall()
    assert before == after


def test_rowid_change_moves_index_entry(populated):
    # INTEGER PRIMARY KEY is an alias for rowid - changing it must not leave
    # a stale entry behind
    populated.update(2, {"id": 22})
    ids = rtree_ids(populated.db)
    assert 2 not in ids
    assert 22 in ids
    assert {r["id"] for r in populated.rows_within_bbox((4.0, 6.0, 4.0, 6.0))} == {1, 22}


def test_delete_removes_index_entry(populated):
    populated.delete(1)
    assert 1 not in rtree_ids(populated.db)
    populated.delete_where("id >= 3")
    assert rtree_ids(populated.db) == {2}


def test_upsert_syncs_index(populated):
    populated.upsert(
        {"id": 1, "minx": 0.0, "maxx": 10.0, "miny": 0.0, "maxy": 10.0}, pk="id"
    )
    assert rtree_ids(populated.db) == {1, 2, 3, 5}


# --- rejection at the write boundary -----------------------------------------

@pytest.mark.parametrize(
    "row",
    [
        # inverted on x, inverted on y
        {"id": 90, "minx": 9.0, "maxx": 1.0, "miny": 0.0, "maxy": 1.0},
        {"id": 91, "minx": 0.0, "maxx": 1.0, "miny": 9.0, "maxy": 1.0},
        # above the 32-bit float range
        {"id": 92, "minx": 0.0, "maxx": 1e39, "miny": 0.0, "maxy": 1.0},
        {"id": 93, "minx": -1e39, "maxx": 0.0, "miny": 0.0, "maxy": 1.0},
        # below the smallest normal 32-bit float
        {"id": 94, "minx": 1e-320, "maxx": 1.0, "miny": 0.0, "maxy": 1.0},
        {"id": 95, "minx": 0.0, "maxx": 1.0, "miny": -1e-320, "maxy": 1.0},
    ],
)
def test_insert_rejects_unindexable_rectangles(populated, row):
    with pytest.raises(sqlite3.IntegrityError):
        populated.insert(row)
    assert row["id"] not in rtree_ids(populated.db)
    # the row itself was not written either
    assert populated.count_where("id = ?", [row["id"]]) == 0


def test_update_rejects_inverted_rectangle(populated):
    with pytest.raises(sqlite3.IntegrityError):
        populated.update(1, {"minx": 99.0})
    # index entry untouched
    row = populated.db.execute(
        "SELECT minx, maxx FROM rtree_boxes WHERE id = 1"
    ).fetchone()
    assert row == (0.0, 10.0)


def test_update_rejects_out_of_range_coordinate(populated):
    with pytest.raises(sqlite3.IntegrityError):
        populated.update(1, {"maxx": 1e39})


def test_nan_bound_from_python_is_stored_as_null(populated):
    # Python's sqlite3 binds float("nan") as NULL, so a NaN can never be
    # indexed - it lands in the NULL path and is simply skipped
    populated.insert(
        {"id": 96, "minx": float("nan"), "maxx": 1.0, "miny": 0.0, "maxy": 1.0}
    )
    assert populated.get(96)["minx"] is None
    assert 96 not in rtree_ids(populated.db)


# --- query semantics ---------------------------------------------------------

def test_intersects_matches_overlapping_and_touching(populated):
    ids = {
        r["id"]
        for r in populated.rows_within_bbox((4.0, 6.0, 4.0, 6.0))
    }
    # big box, point, and the zero-area point at origin via? no - only a/point
    assert ids == {1, 2}


def test_intersects_includes_touching_edge(populated):
    ids = {
        r["id"] for r in populated.rows_within_bbox((10.0, 11.0, 0.0, 5.0))
    }
    assert ids == {1}


def test_contains_requires_enclosing_rectangle(populated):
    ids = {
        r["id"]
        for r in populated.rows_within_bbox((0.0, 10.0, 0.0, 10.0), mode="contains")
    }
    assert ids == {1}


def test_contains_includes_zero_area_row_when_identical(populated):
    ids = {
        r["id"]
        for r in populated.rows_within_bbox((5.0, 5.0, 5.0, 5.0), mode="contains")
    }
    assert ids == {1, 2}


def test_contains_excludes_rows_inside_the_query(populated):
    # The query is bigger than every row, so no row's box encloses it
    ids = {
        r["id"]
        for r in populated.rows_within_bbox((0.0, 100.0, 0.0, 100.0), mode="contains")
    }
    assert ids == set()


def test_negative_coordinates_query(populated):
    ids = {
        r["id"] for r in populated.rows_within_bbox((-15.0, -12.0, -15.0, -12.0))
    }
    assert ids == {3}


def test_null_rows_never_match(populated):
    ids = {
        r["id"]
        for r in populated.rows_within_bbox((-1000.0, 1000.0, -1000.0, 1000.0))
    }
    assert 4 not in ids


def test_invalid_query_bounds_raise(populated):
    for bad in (
        None,
        (1, 2, 3),
        (9.0, 1.0, 0.0, 1.0),
        (0.0, 1.0, 9.0, 1.0),
        (float("nan"), 1.0, 0.0, 1.0),
        (1e39, 2e39, 0.0, 1.0),
        (1e-320, 1.0, 0.0, 1.0),
        ("0", 1.0, 0.0, 1.0),
    ):
        with pytest.raises(InvalidRectangle):
            list(populated.rows_within_bbox(bad))


def test_bad_mode_raises(populated):
    with pytest.raises(ValueError):
        list(populated.rows_within_bbox((0.0, 1.0, 0.0, 1.0), mode="nearby"))


def test_query_requires_index(fresh_db):
    table = make_table(fresh_db)
    with pytest.raises(ValueError):
        list(table.rows_within_bbox((0.0, 1.0, 0.0, 1.0)))


def test_limit_offset_order_by_and_select(populated):
    rows = list(
        populated.rows_within_bbox(
            (-100.0, 100.0, -100.0, 100.0),
            order_by="id desc",
            limit=2,
            select="id",
        )
    )
    assert rows == [{"id": 5}, {"id": 3}]
    rows = list(
        populated.rows_within_bbox(
            (-100.0, 100.0, -100.0, 100.0),
            order_by="id",
            limit=2,
            offset=2,
            select="id",
        )
    )
    assert [r["id"] for r in rows] == [3, 5]


def test_candidate_superset_can_over_select(fresh_db):
    # A value whose outward-rounded 32-bit bounds widen the box: the RTree
    # candidate set is a superset, exact re-check narrows it back down
    table = make_table(fresh_db)
    table.enable_rtree(list(RTREE_COLUMNS))
    table.insert(
        {"id": 1, "minx": 0.1, "maxx": 0.1, "miny": 0.1, "maxy": 0.1}
    )
    stored_minx = fresh_db.execute(
        "SELECT minx FROM rtree_boxes"
    ).fetchone()[0]
    assert stored_minx < 0.1  # outward rounding widened the box
    # Query box that the rounded box intersects but the exact box does not
    probe = (stored_minx, stored_minx, 0.1, 0.1)
    candidate = {r["id"] for r in table.rows_within_bbox(probe, exact=False)}
    exact = {r["id"] for r in table.rows_within_bbox(probe, exact=True)}
    assert 1 in candidate  # RTree returns the candidate
    assert exact == set()  # exact comparison against original values drops it


def test_no_stale_index_entries_after_updates_and_deletes(populated):
    populated.update(2, {"minx": 50.0, "maxx": 60.0, "miny": 50.0, "maxy": 60.0})
    populated.update(3, {"id": 33})
    populated.update(5, {"minx": None})
    populated.delete(1)
    # Independent expectation from the current original table values:
    # rowids of rows whose four coordinates are all non-NULL
    expected_rowids = {
        row[0]
        for row in populated.db.execute(
            "SELECT rowid, minx, maxx, miny, maxy FROM boxes"
        ).fetchall()
        if all(value is not None for value in row[1:])
    }
    ids = rtree_ids(populated.db)
    assert ids == expected_rowids
    # no duplicate entries
    counts = populated.db.execute(
        "SELECT count(*), count(DISTINCT id) FROM rtree_boxes"
    ).fetchone()
    assert counts[0] == counts[1]


# --- independent cross-check against the original table ----------------------

def _boundary_values():
    # Values generated with the standard library only, clustered around the
    # tricky boundaries: 0, +/-1, float32 extrema, negatives, decimals
    values = [0.0, -0.0, 1.0, -1.0, 0.1, -0.1, 0.5, -0.5]
    for exponent in (-100, -50, -10, -1, 1, 10, 20, 30):
        values.extend((math.ldexp(1.0, exponent), -math.ldexp(1.0, exponent)))
    values.extend((F32_MAX, -F32_MAX, F32_MIN_NORMAL, -F32_MIN_NORMAL))
    f32max = struct.unpack("f", struct.pack("f", 3.4e38))[0]
    values.extend((f32max, -f32max, f32max * 0.5, -f32max * 0.5))
    return values


def test_randomized_candidates_never_miss_and_exact_never_over_selects(fresh_db):
    rng = random.Random(416)
    boundary = _boundary_values()
    table = make_table(fresh_db)
    table.enable_rtree(list(RTREE_COLUMNS))
    rows = []
    next_id = 1
    for _ in range(120):
        x1, x2 = rng.sample(boundary, 2)
        y1, y2 = rng.sample(boundary, 2)
        row = {
            "id": next_id,
            "minx": min(x1, x2),
            "maxx": max(x1, x2),
            "miny": min(y1, y2),
            "maxy": max(y1, y2),
        }
        # keep everything inside the accepted 32-bit float range
        if any(abs(v) > F32_MAX for v in (row["minx"], row["maxx"], row["miny"], row["maxy"])):
            continue
        rows.append(row)
        next_id += 1
    table.insert_all(rows)
    stored_rows = list(table.rows)

    queries = []
    for _ in range(60):
        x1, x2 = rng.sample(boundary, 2)
        y1, y2 = rng.sample(boundary, 2)
        queries.append((min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)))

    for bounds in queries:
        for mode in ("intersects", "contains"):
            reference = reference_matches(stored_rows, bounds, mode)
            exact = {
                r["id"] for r in table.rows_within_bbox(bounds, mode=mode)
            }
            candidate = {
                r["id"]
                for r in table.rows_within_bbox(bounds, mode=mode, exact=False)
            }
            # Exact results match the independent reference exactly
            assert exact == reference, (bounds, mode, exact, reference)
            # Candidates are a superset: nothing true is missed
            assert reference.issubset(candidate), (bounds, mode, reference, candidate)


def test_rtree_uses_outward_float32_rounding(fresh_db):
    table = make_table(fresh_db)
    table.enable_rtree(list(RTREE_COLUMNS))
    table.insert({"id": 1, "minx": 0.1, "maxx": 0.1, "miny": 0.1, "maxy": 0.1})
    minx, maxx = fresh_db.execute(
        "SELECT minx, maxx FROM rtree_boxes"
    ).fetchone()
    # stored box is 32-bit floats and strictly encloses the 64-bit value
    assert minx < 0.1 < maxx
    assert isinstance(minx, float) and isinstance(maxx, float)


def test_exact_match_touches_float32_boundary(fresh_db):
    # Two boxes whose 64-bit edges are identical must intersect even though
    # the rounded representations differ - the exact check uses original values
    table = make_table(fresh_db)
    table.enable_rtree(list(RTREE_COLUMNS))
    edge = 123456.789
    table.insert_all(
        [
            {"id": 1, "minx": 0.0, "maxx": edge, "miny": 0.0, "maxy": 1.0},
            {"id": 2, "minx": edge, "maxx": edge + 1, "miny": 0.0, "maxy": 1.0},
        ]
    )
    ids = {r["id"] for r in table.rows_within_bbox((edge, edge, 0.0, 1.0))}
    assert ids == {1, 2}


# --- rebuild, disable and multiple indexes -----------------------------------

def test_disable_rtree_removes_everything(populated):
    populated.disable_rtree()
    assert not populated.db["rtree_boxes"].exists()
    assert populated.detect_rtree() == []
    assert populated.triggers == []
    assert populated.db.execute(
        "SELECT count(*) FROM _rtree_columns"
    ).fetchone()[0] == 0


def test_disable_rtree_is_idempotent(fresh_db):
    table = make_table(fresh_db)
    table.disable_rtree()
    table.disable_rtree(index_name="rtree_boxes")


def test_rebuild_rtree_reindexes_and_recreates_triggers(populated):
    # Simulate a table rebuild (as transform() does): triggers are dropped but
    # rowids and the index entries survive
    populated.db.executescript(
        "DROP TRIGGER rtree_boxes_bi; DROP TRIGGER rtree_boxes_bu; "
        "DROP TRIGGER rtree_boxes_ai; DROP TRIGGER rtree_boxes_au; "
        "DROP TRIGGER rtree_boxes_ad;"
    )
    populated.update(2, {"minx": 50.0, "maxx": 60.0, "miny": 50.0, "maxy": 60.0})
    # stale: the update was not applied to the index
    assert populated.db.execute(
        "SELECT minx FROM rtree_boxes WHERE id = 2"
    ).fetchone()[0] == 5.0
    populated.rebuild_rtree()
    assert len(populated.triggers) == 5
    assert populated.db.execute(
        "SELECT minx FROM rtree_boxes WHERE id = 2"
    ).fetchone()[0] == 50.0
    # triggers work again - keep a valid rectangle when changing one edge
    populated.update(2, {"minx": 40.0, "maxx": 60.0, "miny": 50.0, "maxy": 60.0})
    assert populated.db.execute(
        "SELECT minx FROM rtree_boxes WHERE id = 2"
    ).fetchone()[0] == 40.0


def test_rebuild_rtree_without_index_raises(fresh_db):
    table = make_table(fresh_db)
    with pytest.raises(ValueError):
        table.rebuild_rtree()


def test_multiple_rtree_indexes_coexist(fresh_db):
    table = make_table(fresh_db)
    fresh_db.executescript(
        "ALTER TABLE boxes ADD COLUMN alt_minx REAL; "
        "ALTER TABLE boxes ADD COLUMN alt_maxx REAL; "
        "ALTER TABLE boxes ADD COLUMN alt_miny REAL; "
        "ALTER TABLE boxes ADD COLUMN alt_maxy REAL;"
    )
    table.enable_rtree(list(RTREE_COLUMNS))
    table.enable_rtree(
        ["alt_minx", "alt_maxx", "alt_miny", "alt_maxy"],
        index_name="rtree_boxes_alt",
    )
    assert set(table.detect_rtree()) == {"rtree_boxes", "rtree_boxes_alt"}
    table.insert(
        {
            "id": 1,
            "minx": 0.0,
            "maxx": 1.0,
            "miny": 0.0,
            "maxy": 1.0,
            "alt_minx": 2.0,
            "alt_maxx": 3.0,
            "alt_miny": 2.0,
            "alt_maxy": 3.0,
        }
    )
    assert rtree_ids(fresh_db) == {1}
    assert {
        r[0]
        for r in fresh_db.execute("SELECT id FROM rtree_boxes_alt").fetchall()
    } == {1}
    table.disable_rtree(index_name="rtree_boxes_alt")
    assert table.detect_rtree() == ["rtree_boxes"]


def test_quoted_column_names(fresh_db):
    fresh_db.execute(
        'CREATE TABLE q ("weird minx" REAL, "weird maxx" REAL, '
        '"weird miny" REAL, "weird maxy" REAL)'
    )
    table = fresh_db["q"]
    table.enable_rtree(["weird minx", "weird maxx", "weird miny", "weird maxy"])
    table.insert(
        {
            "weird minx": 0.0,
            "weird maxx": 1.0,
            "weird miny": 0.0,
            "weird maxy": 1.0,
        }
    )
    rows = list(
        table.rows_within_bbox((0.0, 1.0, 0.0, 1.0))
    )
    assert len(rows) == 1


# --- CLI ---------------------------------------------------------------------

def test_cli_enable_and_disable_rtree(tmpdir):
    db_path = str(tmpdir / "rtree.db")
    runner = CliRunner()
    result = runner.invoke(cli, ["create-database", db_path])
    assert result.exit_code == 0
    result = runner.invoke(
        cli,
        [
            "create-table",
            db_path,
            "places",
            "id",
            "integer",
            "minx",
            "real",
            "maxx",
            "real",
            "miny",
            "real",
            "maxy",
            "real",
            "--pk",
            "id",
        ],
    )
    assert result.exit_code == 0
    result = runner.invoke(
        cli,
        ["enable-rtree", db_path, "places", "minx", "maxx", "miny", "maxy"],
    )
    assert result.exit_code == 0, result.output
    db = sqlite_utils.Database(db_path)
    assert db["rtree_places"].virtual_table_using == "RTREE"
    assert len(db.table("places").triggers) == 5
    result = runner.invoke(cli, ["disable-rtree", db_path, "places"])
    assert result.exit_code == 0
    db = sqlite_utils.Database(db_path)
    assert not db["rtree_places"].exists()
    assert db.table("places").triggers == []


def test_cli_enable_rtree_rejects_bad_table(tmpdir):
    db_path = str(tmpdir / "rtree.db")
    runner = CliRunner()
    runner.invoke(cli, ["create-database", db_path])
    result = runner.invoke(
        cli, ["enable-rtree", db_path, "nope", "a", "b", "c", "d"]
    )
    assert result.exit_code == 1
    assert "nope" in result.output


# --- additional edge cases ---------------------------------------------------

def test_enable_rtree_on_empty_table_then_insert(fresh_db):
    table = make_table(fresh_db)
    table.enable_rtree(list(RTREE_COLUMNS))
    table.insert({"id": 1, "minx": 0.0, "maxx": 1.0, "miny": 0.0, "maxy": 1.0})
    assert rtree_ids(fresh_db) == {1}


def test_enable_rtree_on_integer_columns(fresh_db):
    fresh_db.execute(
        "CREATE TABLE tiles (id INTEGER PRIMARY KEY, x0 INTEGER, x1 INTEGER, "
        "y0 INTEGER, y1 INTEGER)"
    )
    table = fresh_db["tiles"]
    table.enable_rtree(["x0", "x1", "y0", "y1"])
    table.insert_all(
        [
            {"id": 1, "x0": 0, "x1": 10, "y0": 0, "y1": 10},
            {"id": 2, "x0": 20, "x1": 30, "y0": 20, "y1": 30},
        ]
    )
    assert {
        r["id"] for r in table.rows_within_bbox((5, 25, 5, 25))
    } == {1, 2}
    assert {
        r["id"]
        for r in table.rows_within_bbox((5, 25, 5, 25), mode="contains")
    } == set()
    assert fresh_db["rtree_tiles"].columns_dict == {
        "id": int,
        "minx": float,
        "maxx": float,
        "miny": float,
        "maxy": float,
    }


def test_strict_table_supported(fresh_db):
    table = make_table(fresh_db, strict=True)
    table.enable_rtree(list(RTREE_COLUMNS))
    table.insert({"id": 1, "minx": 0.0, "maxx": 1.0, "miny": 0.0, "maxy": 1.0})
    assert rtree_ids(fresh_db) == {1}
