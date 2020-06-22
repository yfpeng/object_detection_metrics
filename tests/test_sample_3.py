import json
from pathlib import Path

from podm.podm import get_pascal_voc_metrics
from tests.test_utils import load_data, test_results

RESULT0_3 = {
    'person': {
        'AP': 0.245687
    }
}


if __name__ == '__main__':
    dir = Path('sample_3')
    gt_BoundingBoxes = load_data(dir / 'groundtruths.json')
    pd_BoundingBoxes = load_data(dir / 'detections.json')

    RESULT0_5 = json.load(open(dir / 'expected0_5.json'))
    results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)
    test_results(RESULT0_5, results, 'AP')
    test_results(RESULT0_5, results, 'precision')
    test_results(RESULT0_5, results, 'recall')

    results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .3)
    test_results(RESULT0_3, results, 'AP')
