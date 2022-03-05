import json
from pathlib import Path

import numpy as np

from helpers.utils import assert_results
from podm import get_pascal_voc_metrics, MetricPerClass, load_data


def test_sample_2(tests_dir):
    dir = tests_dir / 'sample_2'
    gt_BoundingBoxes = load_data(dir / 'groundtruths.json')
    pd_BoundingBoxes = load_data(dir / 'detections.json')

    RESULT0_5 = json.load(open(dir / 'expected0_5.json'))
    results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)
    assert_results(results, RESULT0_5, 'ap')
    assert_results(results, RESULT0_5, 'precision')
    assert_results(results, RESULT0_5, 'recall')
    assert_results(results, RESULT0_5, 'tp')
    assert_results(results, RESULT0_5, 'fp')
    assert_results(results, RESULT0_5, 'num_groundtruth')
    assert_results(results, RESULT0_5, 'num_detection')

    mAP = MetricPerClass.mAP(results)
    assert np.isclose(RESULT0_5['mAP'], mAP, 1e-3), mAP


if __name__ == '__main__':
    test_sample_2(Path(__file__).parent)
