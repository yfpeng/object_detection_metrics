import json
from pathlib import Path

from podm.podm import get_pascal_voc_metrics
from podm.visualize import plot_precision_recall_curve_all
from tests.test_utils import load_data

if __name__ == '__main__':
    dir = Path('sample_2')
    gt_BoundingBoxes = load_data(dir / 'groundtruths.json')
    pd_BoundingBoxes = load_data(dir / 'detections.json')

    results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)
    plot_precision_recall_curve_all(results, dir / 'plots', show_interpolated_precision=True)
