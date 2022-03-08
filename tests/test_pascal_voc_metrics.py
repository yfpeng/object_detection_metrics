import json
import math
from pathlib import Path

from helpers.utils import assert_results
from podm import pcoco_decoder
from podm.metrics import get_pascal_voc_metrics, MetricPerClass


def _get_dataset_helper(dir):
    with open(dir / 'groundtruths_coco.json') as fp:
        gold_dataset = pcoco_decoder.load_object_detection(fp)
    with open(dir / 'detections_coco.json') as fp:
        pred_dataset = pcoco_decoder.load_object_detection_result(fp, gold_dataset)

    RESULT0_5 = json.load(open(dir / 'expected0_5.json'))
    expects = {}
    for cat in RESULT0_5.keys():
        category = gold_dataset.get_category_name(cat)
        if category is not None:
            expects[gold_dataset.get_category_name(cat).id] = RESULT0_5[cat]
    expects.update(**RESULT0_5)
    return gold_dataset, pred_dataset, expects


def test_sample2(tests_dir):
    dir = tests_dir / 'sample_2'
    gold_dataset, pred_dataset, expects = _get_dataset_helper(dir)

    results = get_pascal_voc_metrics(gold_dataset.bboxes, pred_dataset.bboxes, .5)
    assert_results(results, expects, 'ap')
    assert_results(results, expects, 'precision')
    assert_results(results, expects, 'recall')
    assert_results(results, expects, 'tp')
    assert_results(results, expects, 'fp')
    assert_results(results, expects, 'num_groundtruth')
    assert_results(results, expects, 'num_detection')

    mAP = MetricPerClass.mAP(results)
    assert math.isclose(expects['mAP'], mAP, rel_tol=1e-3), '{} vs {}'.format(expects['mAP'], mAP)


def test_sample3(tests_dir):
    dir = tests_dir / 'sample_3'
    gold_dataset, pred_dataset, expects = _get_dataset_helper(dir)

    results = get_pascal_voc_metrics(gold_dataset.bboxes, pred_dataset.bboxes, .5)
    assert_results(results, expects, 'ap')
    assert_results(results, expects, 'precision')
    assert_results(results, expects, 'recall')

    RESULT0_3 = {
        'person': {
            'ap': 0.245687
        }
    }
    results = get_pascal_voc_metrics(gold_dataset.bboxes, pred_dataset.bboxes, .3)
    assert_results(results, RESULT0_3, 'ap')


if __name__ == '__main__':
    test_sample2(Path(__file__).parent)
