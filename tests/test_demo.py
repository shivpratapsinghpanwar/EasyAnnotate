"""easyannotate demo: the first minute has to show the whole loop."""

from __future__ import annotations

import tempfile
from pathlib import Path

from data_factory.annotate import attach, demo, images, review
from data_factory.annotate.store import Store


def test_the_demo_opens_with_labelled_predicted_and_untouched_images(tmp_path):
    store = Store(tmp_path / "annotations.db")
    try:
        built = demo.build(tmp_path / "demo", images=12)
        assert built["fresh"]
        summary = attach.open_folder(store, tmp_path / "demo", name="demo")
        project = summary["project"]
        with tempfile.TemporaryDirectory() as holding:
            run = demo.write_predictions(built["predictions"], Path(holding) / "run")
            report = review.import_predictions(store, project, run)

        counts = images.counts(store, project)
        assert summary["shapes_imported"] > 0
        assert counts["reviewed"] == 2          # labelled by hand
        assert report["matched"] == 6 and counts["predicted"] == 6
        assert report["unmatched_files"] == 0 and report["ambiguous_files"] == 0
        assert counts["todo"] == 4

        # A second run reopens the same folder rather than rebuilding it.
        assert demo.build(tmp_path / "demo", images=12)["fresh"] is False
    finally:
        store.close()
