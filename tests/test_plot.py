from pathlib import Path

from podm import pcoco_decoder
from podm.metrics import get_pascal_voc_metrics
from podm.visualize import plot_precision_recall_curve_all


def test_plot(tests_dir, tmp_path):
    with open(tests_dir / 'sample_2' / 'groundtruths_coco.json') as fp:
        gold_dataset = pcoco_decoder.load_object_detection(fp)
    with open(tests_dir / 'sample_2' / 'detections_coco.json') as fp:
        pred_dataset = pcoco_decoder.load_object_detection_result(fp, gold_dataset)
    results = get_pascal_voc_metrics(gold_dataset.bboxes(True), pred_dataset.bboxes(True), .5)
    plot_precision_recall_curve_all(results, tmp_path, show_interpolated_precision=True)

