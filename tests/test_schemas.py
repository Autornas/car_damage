import pytest
from pydantic import ValidationError

from backend.schemas import BoxIn


def test_box_in_accepts_coordinate_only_correction():
    box = BoxIn(
        x_min=1028.4,
        y_min=120.2,
        x_max=1300.6,
        y_max=875.8,
    )

    assert box.x_min == 1028.4
    assert box.y_max == 875.8


def test_box_in_rejects_invalid_coordinate_ordering():
    with pytest.raises(ValidationError, match="x_min must be < x_max"):
        BoxIn(x_min=10, y_min=20, x_max=5, y_max=30)
