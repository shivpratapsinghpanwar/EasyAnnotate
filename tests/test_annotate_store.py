"""The annotation store: schema, ownership, and class-index stability.

The class-index tests are the point of this file. They encode a rule that is
invisible at runtime -- nothing errors when indices shift, the exported files
simply start meaning something else -- so a test is the only thing standing
between a rename and a silently corrupted dataset.
"""

from __future__ import annotations

import sqlite3

import pytest

from data_factory.annotate import labels, projects
from data_factory.annotate.store import Conflict, NotFound, Store, StoreError, valid_id


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "annotations.db")
    yield s
    s.close()


@pytest.fixture()
def project(store):
    return projects.create(store, "Doors")


# ------------------------------------------------------------------- schema

def test_opening_twice_does_not_remigrate(tmp_path):
    from data_factory.annotate.store import SCHEMA_VERSION

    first = Store(tmp_path / "a.db")
    first.close()
    second = Store(tmp_path / "a.db")
    # Against the constant, not a literal: a hard-coded version here just goes
    # stale the next time a rung is added, which tells you nothing.
    assert second.one("PRAGMA user_version")[0] == SCHEMA_VERSION
    second.close()


def test_a_newer_database_is_refused_not_guessed_at(tmp_path):
    path = tmp_path / "future.db"
    Store(path).close()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version=99")
    conn.commit()
    conn.close()

    with pytest.raises(StoreError, match="schema version 99"):
        Store(path)


def test_foreign_keys_are_enforced(store, project):
    # A shape referencing a class index that was never created must not insert.
    with pytest.raises(StoreError):
        with store.write() as conn:
            conn.execute(
                "INSERT INTO image (id, project_id, rel_path, created_at, updated_at)"
                " VALUES ('i', ?, 'a.jpg', 0, 0)",
                (project.id,),
            )
            conn.execute(
                "INSERT INTO shape (id, image_id, project_id, label_idx, xmin, ymin,"
                " xmax, ymax, created_at) VALUES ('s', 'i', ?, 7, 0, 0, 1, 1, 0)",
                (project.id,),
            )


def test_ids_are_opaque_not_sequential(store):
    first = projects.create(store, "one")
    second = projects.create(store, "two")
    assert valid_id(first.id) and valid_id(second.id)
    assert first.id != second.id


# ---------------------------------------------------------------- ownership

def test_a_project_is_invisible_to_another_owner(store):
    mine = projects.create(store, "mine", owner_id="alice")
    with pytest.raises(NotFound):
        projects.get(store, mine.id, owner_id="bob")


def test_listing_only_shows_your_own(store):
    projects.create(store, "a", owner_id="alice")
    projects.create(store, "b", owner_id="alice")
    projects.create(store, "c", owner_id="bob")
    assert len(projects.listing(store, "alice")) == 2
    assert len(projects.listing(store, "bob")) == 1


def test_a_traversal_id_is_refused(store):
    with pytest.raises(NotFound):
        projects.get(store, "../../etc/passwd")


def test_deleting_a_project_takes_its_labels_with_it(store, project):
    labels.ensure(store, project, "Open")
    projects.delete(store, project)
    remaining = store.query("SELECT * FROM label WHERE project_id = ?", (project.id,))
    assert remaining == []


# ------------------------------------------------------------------- labels

def test_a_label_is_created_by_using_it(store, project):
    label = labels.ensure(store, project, "Open")
    assert label.idx == 0
    assert label.colour.startswith("#")
    # Idempotent: the editor calls this on every save.
    assert labels.ensure(store, project, "Open").idx == 0


def test_names_are_whitespace_normalised(store, project):
    first = labels.ensure(store, project, "front door")
    second = labels.ensure(store, project, "  front   door  ")
    assert first.idx == second.idx
    assert second.name == "front door"


def test_indices_are_allocated_in_creation_order_not_alphabetical(store, project):
    # This is the crux. Sorted order would put Closed first.
    assert labels.ensure(store, project, "Open").idx == 0
    assert labels.ensure(store, project, "Closed").idx == 1
    assert labels.class_list(store, project) == ["Open", "Closed"]


def test_renaming_does_not_move_the_index(store, project):
    labels.ensure(store, project, "Open")
    labels.ensure(store, project, "Cosed")
    labels.rename(store, project, 1, "Closed")
    assert labels.class_list(store, project) == ["Open", "Closed"]
    assert labels.get(store, project, 1).name == "Closed"


def test_renaming_onto_another_live_name_is_refused(store, project):
    labels.ensure(store, project, "Open")
    labels.ensure(store, project, "Closed")
    with pytest.raises(StoreError, match="already called"):
        labels.rename(store, project, 1, "Open")


def test_retiring_keeps_the_slot_so_later_indices_do_not_move(store, project):
    labels.ensure(store, project, "Open")     # 0
    labels.ensure(store, project, "Handle")   # 1
    labels.ensure(store, project, "Closed")   # 2
    labels.retire(store, project, 1)

    assert [l.idx for l in labels.listing(store, project)] == [0, 2]
    # But the export list still positions Closed at 2.
    assert labels.class_list(store, project) == ["Open", "Handle", "Closed"]


def test_a_retired_index_is_never_handed_out_again(store, project):
    labels.ensure(store, project, "Open")     # 0
    labels.ensure(store, project, "Handle")   # 1
    labels.retire(store, project, 1)
    fresh = labels.ensure(store, project, "Hinge")
    assert fresh.idx == 2, "the counter must not rewind into a retired slot"


def test_a_retired_name_can_be_used_again_as_a_new_index(store, project):
    labels.ensure(store, project, "Open")
    labels.retire(store, project, 0)
    again = labels.ensure(store, project, "Open")
    assert again.idx == 1
    assert labels.class_list(store, project) == ["Open", "Open"]


def test_restoring_a_retired_label_is_refused_if_the_name_was_reused(store, project):
    labels.ensure(store, project, "Open")
    labels.retire(store, project, 0)
    labels.ensure(store, project, "Open")
    with pytest.raises(StoreError, match="in use"):
        labels.restore(store, project, 0)


def test_an_empty_name_is_refused(store, project):
    with pytest.raises(StoreError):
        labels.ensure(store, project, "   ")


def test_adopting_an_existing_class_order_preserves_positions(store, project):
    labels.adopt(store, project, ["person", "car", "dog"])
    assert labels.class_list(store, project) == ["person", "car", "dog"]
    # New labels continue after the adopted ones.
    assert labels.ensure(store, project, "cat").idx == 3


def test_a_class_order_cannot_be_adopted_over_existing_labels(store, project):
    labels.ensure(store, project, "Open")
    with pytest.raises(StoreError, match="before any label"):
        labels.adopt(store, project, ["a", "b"])


# ----------------------------------------------- the bug this table replaces

def test_adding_a_class_mid_project_leaves_earlier_indices_untouched(store, project):
    """The Monday/Friday case, stated as an assertion.

    Under the old ``sorted(found)`` scheme, adding 'Handle' would renumber
    'Open' from 1 to 2, and every .txt exported before that would start
    meaning 'Handle'.
    """
    labels.ensure(store, project, "Closed")
    labels.ensure(store, project, "Open")
    monday = labels.class_list(store, project)

    labels.ensure(store, project, "Handle")
    friday = labels.class_list(store, project)

    assert monday == ["Closed", "Open"]
    assert friday[: len(monday)] == monday, "existing class indices must not move"
    assert friday.index("Open") == monday.index("Open")


def test_alphabetically_late_first_class_stays_at_zero(store, project):
    # 'zebra' sorts last but was drawn first, so it owns index 0 forever.
    labels.ensure(store, project, "zebra")
    labels.ensure(store, project, "aardvark")
    assert labels.class_list(store, project)[0] == "zebra"


# --------------------------------------------------- merging duplicate classes

def test_case_variant_classes_are_reported_not_folded(store, project):
    """The real door set has Close/close/Open/open. Importing them faithfully
    is right; folding them silently would change somebody's data."""
    for name in ("Close", "Open", "close", "open"):
        labels.ensure(store, project, name)

    found = labels.variants(store, project)
    assert set(found) == {"close", "open"}
    assert [l.idx for l in found["close"]] == [0, 2]


def test_distinct_classes_are_not_reported_as_variants(store, project):
    labels.ensure(store, project, "Open")
    labels.ensure(store, project, "Closed")
    assert labels.variants(store, project) == {}


def test_merging_moves_the_shapes_and_retires_the_source(store, project):
    from data_factory.annotate import images, shapes
    from data_factory.core.annotations.model import Shape

    image = images.add(store, project, "a.jpg", width=100, height=100)
    shapes.replace_all(
        store, project, image,
        [Shape(label="Close", bbox=(1, 1, 10, 10)),
         Shape(label="close", bbox=(20, 20, 30, 30)),
         Shape(label="close", bbox=(40, 40, 50, 50))],
    )
    assert shapes.label_histogram(store, project) == {"Close": 1, "close": 2}

    moved = labels.merge(store, project, 1, 0)   # close -> Close

    assert moved == 2
    assert shapes.label_histogram(store, project) == {"Close": 3}
    assert [l.name for l in labels.listing(store, project)] == ["Close"]
    assert not labels.get(store, project, 1).is_live


def test_merging_does_not_move_any_class_index(store, project):
    """The merged-away index is retired, so it keeps its slot."""
    labels.ensure(store, project, "Close")   # 0
    labels.ensure(store, project, "close")   # 1
    labels.ensure(store, project, "Open")    # 2

    before = labels.class_list(store, project)
    labels.merge(store, project, 1, 0)
    after = labels.class_list(store, project)

    assert before == after == ["Close", "close", "Open"]
    assert after.index("Open") == 2, "a later class shifted"


def test_merging_a_class_into_itself_is_refused(store, project):
    labels.ensure(store, project, "Open")
    with pytest.raises(StoreError, match="into itself"):
        labels.merge(store, project, 0, 0)


def test_merging_into_a_retired_class_is_refused(store, project):
    labels.ensure(store, project, "Open")
    labels.ensure(store, project, "Close")
    labels.retire(store, project, 0)
    with pytest.raises(StoreError, match="retired"):
        labels.merge(store, project, 1, 0)


def test_merging_an_unknown_class_is_refused(store, project):
    labels.ensure(store, project, "Open")
    with pytest.raises(NotFound):
        labels.merge(store, project, 9, 0)
