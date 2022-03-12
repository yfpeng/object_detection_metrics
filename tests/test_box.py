import pytest

from podm import box
from podm.box import Box
import math


def test_iou():
    box1 = Box.of_box(0., 0., 10., 10.)
    box2 = Box.of_box(1., 1., 11., 11.)
    assert math.isclose(box.intersection_over_union(box1, box2), 0.680672268908, rel_tol=1e-6)

    box1 = Box.of_box(0. / 100, 0. / 100, 10. / 100, 10. / 100)
    box2 = Box.of_box(1. / 100, 1. / 100, 11. / 100, 11. / 100)
    assert math.isclose(box.intersection_over_union(box1, box2), 0.680672268908, rel_tol=1e-6)

    # no overlap
    box1 = Box.of_box(0., 0., 10., 10.)
    box2 = Box.of_box(12., 12., 22., 22.)
    assert box.intersection_over_union(box1, box2) == 0

    box1 = Box.of_box(0., 0., 2., 2.)
    box2 = Box.of_box(1., 1., 3., 3.)
    assert math.isclose(box.intersection_over_union(box1, box2), 0.142857142857, rel_tol=1e-6)


def test_union():
    box1 = Box.of_box(0., 0., 10., 10.)
    box2 = Box.of_box(1., 1., 11., 11.)
    assert box.union(box1, box2) == Box.of_box(0, 0, 11, 11)
    assert box.union_areas(box1, box2) == 119


def test_box():
    box = Box.of_box(0., 0., 10., 10.)
    assert box.width == 10
    assert box.height == 10
    assert box.area == 100
    assert box == Box.of_box(0, 0, 10, 10)
    assert box != Box.of_box(0, 0, 11, 11)
    assert box != 1

    with pytest.raises(AssertionError):
        Box.of_box(0., 0., -1, 10.)
