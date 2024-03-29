import json
import math
from pathlib import Path

from helpers.utils import assert_results
from podm import coco_decoder
from podm.metrics import get_pascal_voc_metrics, MetricPerClass, get_bounding_boxes


def _get_dataset_helper(sample_dir):
    with open(sample_dir / 'groundtruths_coco.json') as fp:
        gold_dataset = coco_decoder.load_true_object_detection_dataset(fp)
    with open(sample_dir / 'detections_coco.json') as fp:
        pred_dataset = coco_decoder.load_pred_object_detection_dataset(fp, gold_dataset)
    RESULT0_5 = json.load(open(sample_dir / 'expected0_5.json'))
    return get_bounding_boxes(gold_dataset), get_bounding_boxes(pred_dataset), RESULT0_5


def test_sample2(sample_dir):
    gold_dataset, pred_dataset, expects = _get_dataset_helper(sample_dir)

    results = get_pascal_voc_metrics(gold_dataset, pred_dataset, .5)
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

    results = get_pascal_voc_metrics(gold_dataset, pred_dataset, .5)
    assert_results(results, expects, 'ap')
    assert_results(results, expects, 'precision')
    assert_results(results, expects, 'recall')

    RESULT0_3 = {
        'person': {
            'ap': 0.245687
        }
    }
    results = get_pascal_voc_metrics(gold_dataset, pred_dataset, .3)
    assert_results(results, RESULT0_3, 'ap')


if __name__ == '__main__':
    test_sample2(Path(__file__).parent)
