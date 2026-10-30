import json

from backend import dataset_merge


def _write_source(source_dir, filename, box, split="train"):
    images_dir = source_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / filename).write_bytes(source_dir.name.encode("utf-8"))
    annotations_path = source_dir / "annotations.json"
    annotations = (
        json.loads(annotations_path.read_text(encoding="utf-8"))
        if annotations_path.is_file()
        else {}
    )
    annotations[filename] = {"split": split, "boxes": [box]}
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")


def test_merge_includes_registered_sources_and_prioritizes_corrections(tmp_path, monkeypatch):
    pilot = tmp_path / "phase2_pilot"
    external = tmp_path / "cardd_combined"
    corrections = tmp_path / "corrections"
    original_box = {"x_min": 0, "y_min": 0, "x_max": 4, "y_max": 4}
    corrected_box = {"x_min": 1, "y_min": 1, "x_max": 5, "y_max": 5}
    external_box = {"x_min": 2, "y_min": 2, "x_max": 6, "y_max": 6}

    _write_source(pilot, "pilot.jpg", original_box)
    _write_source(external, "cardd.jpg", external_box)
    _write_source(pilot, "corrected.jpg", original_box, split="val")
    _write_source(corrections, "corrected.jpg", corrected_box)

    monkeypatch.setattr(dataset_merge, "CORRECTIONS_DIR", corrections)
    monkeypatch.setattr(
        dataset_merge, "list_source_dirs", lambda raw_dir: [corrections, external, pilot],
    )
    output_dir = tmp_path / "merged"

    dataset_merge.merge_datasets(output_dir)

    merged = json.loads((output_dir / "annotations.json").read_text(encoding="utf-8"))
    assert set(merged) == {"pilot.jpg", "cardd.jpg", "corrected.jpg"}
    assert merged["cardd.jpg"]["boxes"] == [external_box]
    assert merged["corrected.jpg"]["boxes"] == [corrected_box]
    assert merged["corrected.jpg"]["split"] == "val"
