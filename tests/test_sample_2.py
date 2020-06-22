import json
from pathlib import Path

import numpy as np

from podm.podm import get_pascal_voc_metrics, get_mAP
from tests.test_utils import load_data, test_results

if __name__ == '__main__':
    dir = Path('sample_2')
    gt_BoundingBoxes = load_data(dir / 'groundtruths.json')
    pd_BoundingBoxes = load_data(dir / 'detections.json')

    RESULT0_5 = json.load(open(dir / 'expected0_5.json'))
    results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)
    test_results(RESULT0_5, results, 'AP')
    test_results(RESULT0_5, results, 'precision')
    test_results(RESULT0_5, results, 'recall')
    test_results(RESULT0_5, results, 'tp')
    test_results(RESULT0_5, results, 'fp')
    test_results(RESULT0_5, results, 'Number of ground-truth objects')
    test_results(RESULT0_5, results, 'Number of detected objects')

    mAP = get_mAP(results)
    assert np.isclose(RESULT0_5['mAP'], mAP, 1e-3), mAP
