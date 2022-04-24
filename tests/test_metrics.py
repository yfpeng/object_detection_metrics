import numpy as np

from podm.metrics import calculate_11_points_average_precision, calculate_all_points_average_precision


def test_calculate_11_points_average_precision():
    recall = 0.5
    precision = 1
    ap, r_values, p_values = calculate_11_points_average_precision([recall], [precision])
    assert np.allclose(ap, 0.545, rtol=1e-3, equal_nan=True)
    assert np.allclose(r_values, [0, 0, .1, .2, .3, .4, .5, .5, .6, .7, .8, .9, 1], rtol=1e-1)
    assert np.allclose(p_values, [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], rtol=1e-1)


# def test_calculate_all_points_average_precision():
#     recall = 0.5
#     precision = 1
#     ap, r_values, p_values = calculate_all_points_average_precision([recall], [precision])
#     assert np.allclose(ap, 0.545, rtol=1e-3, equal_nan=True)
#     assert np.allclose(r_values, [1, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1, 0, 0], rtol=1e-1)
#     assert np.allclose(p_values, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0], rtol=1e-1)