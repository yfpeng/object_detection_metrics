from podm import pcoco_decoder
from podm.metrics import get_pascal_voc_metrics, get_bounding_boxes
from podm.visualize import plot_precision_recall_curve_all


def test_plot(tests_dir, tmp_path):
    with open(tests_dir / 'sample_2' / 'groundtruths_coco.json') as fp:
        gold_dataset = pcoco_decoder.load_true_bounding_box_dataset(fp)
    with open(tests_dir / 'sample_2' / 'detections_coco.json') as fp:
        pred_dataset = pcoco_decoder.load_pred_bounding_box_dataset(fp, gold_dataset)
    results = get_pascal_voc_metrics(get_bounding_boxes(gold_dataset), get_bounding_boxes(pred_dataset), .5)
    plot_precision_recall_curve_all(results, tmp_path, show_interpolated_precision=True)

