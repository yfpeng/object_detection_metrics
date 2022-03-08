from pathlib import Path

from podm.metrics import get_pascal_voc_metrics
from podm.utils import load_data
from podm.visualize import plot_precision_recall_curve_all


def test_plot(tests_dir, tmp_path):
    gt_BoundingBoxes = load_data(tests_dir / 'sample_2' / 'groundtruths.json')
    pd_BoundingBoxes = load_data(tests_dir / 'sample_2' / 'detections.json')
    results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)
    plot_precision_recall_curve_all(results, tmp_path, show_interpolated_precision=True)


if __name__ == '__main__':
    dir = Path('sample_2')
    gt_BoundingBoxes = load_data(dir / 'groundtruths.json')
    pd_BoundingBoxes = load_data(dir / 'detections.json')

    results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)
    plot_precision_recall_curve_all(results, dir / 'plots', show_interpolated_precision=True)
