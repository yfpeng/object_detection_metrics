import json
import math
from pathlib import Path

import numpy as np
import pytest

from helpers.utils import assert_results
from podm import get_pascal_voc_metrics, MetricPerClass, load_data, load_data_coco


def test_sample2(tests_dir):
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
    assert math.isclose(RESULT0_5['mAP'], mAP, rel_tol=1e-3), '{} vs {}'.format(RESULT0_5['mAP'], mAP)


def test_sample3(tests_dir):
    dir = tests_dir / 'sample_3'
    gt_BoundingBoxes, pd_BoundingBoxes = load_data_coco(dir / 'groundtruths_coco.json',
                                                        dir / 'detections_coco.json')

    RESULT0_5 = json.load(open(dir / 'expected0_5.json'))
    results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)
    assert_results(results, RESULT0_5, 'ap')
    assert_results(results, RESULT0_5, 'precision')
    assert_results(results, RESULT0_5, 'recall')

    RESULT0_3 = {
        'person': {
            'ap': 0.245687
        }
    }
    results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .3)
    assert_results(results, RESULT0_3, 'ap')


if __name__ == '__main__':
    test_sample3(Path(__file__).parent)
