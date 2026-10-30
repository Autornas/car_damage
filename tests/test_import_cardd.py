import json
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np

from car_damage.data.hashing import compute_phash_from_bytes
from scripts.import_cardd import convert_coco_archives


def _image_bytes(reverse=False):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    for x in range(32):
        value = 255 - x * 8 if reverse else x * 8
        image[:, x] = (value, value, value)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def _write_coco_split(archive, annotation_path, image_root, images, categories, annotations):
    for image_id, filename, image_bytes in images:
        archive.writestr(f"{image_root}/{filename}", image_bytes)
    archive.writestr(
        annotation_path,
        json.dumps({
            "images": [
                {"id": image_id, "file_name": filename, "width": 32, "height": 32}
                for image_id, filename, _ in images
            ],
            "categories": categories,
            "annotations": annotations,
        }),
    )


def test_convert_coco_archives_deduplicates_and_maps_annotations(tmp_path):
    release_zip = tmp_path / "release.zip"
    roboflow_zip = tmp_path / "roboflow.zip"
    shared_image = _image_bytes()
    unique_image = _image_bytes(reverse=True)

    with ZipFile(release_zip, "w", ZIP_DEFLATED) as archive:
        _write_coco_split(
            archive,
            "CarDD_release/CarDD_COCO/annotations/instances_train2017.json",
            "CarDD_release/CarDD_COCO/train2017",
            [(1, "shared.jpg", shared_image)],
            [{"id": 1, "name": "dent"}],
            [{"image_id": 1, "category_id": 1, "bbox": [1, 2, 10, 12]}],
        )
        for split in ("val", "test"):
            _write_coco_split(
                archive,
                f"CarDD_release/CarDD_COCO/annotations/instances_{split}2017.json",
                f"CarDD_release/CarDD_COCO/{split}2017",
                [],
                [{"id": 1, "name": "dent"}],
                [],
            )

    with ZipFile(roboflow_zip, "w", ZIP_DEFLATED) as archive:
        _write_coco_split(
            archive,
            "train/_annotations.coco.json",
            "train",
            [
                (2, "shared.jpg", shared_image),
                (3, "unique.jpg", unique_image),
            ],
            [{"id": 1, "name": "severe-deformation"}],
            [
                {"image_id": 2, "category_id": 1, "bbox": [2, 3, 8, 9]},
                {"image_id": 3, "category_id": 1, "bbox": [4, 5, 6, 7]},
            ],
        )
        for split in ("valid", "test"):
            _write_coco_split(
                archive,
                f"{split}/_annotations.coco.json",
                split,
                [],
                [{"id": 1, "name": "severe-deformation"}],
                [],
            )

    output_dir = tmp_path / "converted"
    stats = convert_coco_archives(release_zip, roboflow_zip, output_dir)
    annotations = json.loads((output_dir / "annotations.json").read_text())

    assert len(annotations) == 2
    assert stats["cardd_release"] == {
        "images": 1, "boxes": 1, "duplicates_skipped": 0,
    }
    assert stats["roboflow_v3"] == {
        "images": 1, "boxes": 1, "duplicates_skipped": 1,
    }

    release_entry = next(entry for name, entry in annotations.items() if name.startswith("cardd_release"))
    assert release_entry["boxes"][0] == {
        "x_min": 1, "y_min": 2, "x_max": 11, "y_max": 14, "damage_type": "dent",
    }
    roboflow_entry = next(entry for name, entry in annotations.items() if name.startswith("roboflow_v3"))
    assert roboflow_entry["boxes"][0]["damage_type"] == "dent"
    assert roboflow_entry["boxes"][0]["severity"] == "severe"


def test_hashing_encoded_bytes_matches_file_hash(tmp_path):
    image_bytes = _image_bytes()
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(image_bytes)

    from car_damage.data.hashing import compute_phash

    assert compute_phash_from_bytes(image_bytes) == compute_phash(image_path)
