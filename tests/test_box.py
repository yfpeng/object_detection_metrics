from podm import Box
import numpy as np


def test_iou():
    box1 = Box(0., 0., 10., 10.)
    box2 = Box(1., 1., 11., 11.)
    assert np.isclose(Box.intersection_over_union(box1, box2), 0.680672268908, 1e-6)

    box1 = Box(0. / 100, 0. / 100, 10. / 100, 10. / 100)
    box2 = Box(1. / 100, 1. / 100, 11. / 100, 11. / 100)
    assert np.isclose(Box.intersection_over_union(box1, box2), 0.680672268908, 1e-6)

    # no overlap
    box1 = Box(0., 0., 10., 10.)
    box2 = Box(12., 12., 22., 22.)
    assert np.isclose(Box.intersection_over_union(box1, box2), 0, 1e-6)

    box1 = Box(0., 0., 2., 2.)
    box2 = Box(1., 1., 3., 3.)
    assert np.isclose(Box.intersection_over_union(box1, box2), 0.142857142857, 1e-6)
