import pytest

from podm import coco_decoder
from podm.metrics import get_pascal_voc_metrics, get_bounding_boxes
from podm.visualize import plot_precision_recall_curve_all

# @pytest.mark.skip()
def test_plot(sample_dir, tmp_path):
    with open(sample_dir / 'groundtruths_coco.json') as fp:
        gold_dataset = coco_decoder.load_true_object_detection_dataset(fp)
    with open(sample_dir / 'detections_coco.json') as fp:
        pred_dataset = coco_decoder.load_pred_object_detection_dataset(fp, gold_dataset)
    results = get_pascal_voc_metrics(get_bounding_boxes(gold_dataset), get_bounding_boxes(pred_dataset), .5)
    plot_precision_recall_curve_all(results, tmp_path, show_interpolated_precision=True)

