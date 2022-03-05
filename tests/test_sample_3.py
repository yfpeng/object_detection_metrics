import json
from pathlib import Path

from podm import get_pascal_voc_metrics
from helpers.utils import load_data, assert_results, load_data_coco

RESULT0_3 = {
    'person': {
        'ap': 0.245687
    }
}


def test_sample3():
    dir = Path('tests/sample_3')
    gt_BoundingBoxes = load_data(dir / 'groundtruths.json')
    pd_BoundingBoxes = load_data(dir / 'detections.json')

    RESULT0_5 = json.load(open(dir / 'expected0_5.json'))
    results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)
    assert_results(results, RESULT0_5, 'ap')
    assert_results(results, RESULT0_5, 'precision')
    assert_results(results, RESULT0_5, 'recall')

    results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .3)
    assert_results(results, RESULT0_3, 'ap')


def test_sample3_coco():
    dir = Path('tests/sample_3')
    gt_BoundingBoxes, pd_BoundingBoxes = load_data_coco(dir / 'groundtruths_coco.json',
                                                        dir / 'detections_coco.json')

    RESULT0_5 = json.load(open(dir / 'expected0_5.json'))
    results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)
    assert_results(results, RESULT0_5, 'ap')
    assert_results(results, RESULT0_5, 'precision')
    assert_results(results, RESULT0_5, 'recall')

    results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .3)
    assert_results(results, RESULT0_3, 'ap')
