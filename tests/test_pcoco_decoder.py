from podm import pcoco_decoder


def test_sample2(tests_dir):
    with open(tests_dir / 'sample_2/groundtruths_coco.json') as fp:
        dataset = pcoco_decoder.load_object_detection(fp)

    assert len(dataset.images) == 85
    assert len(dataset.categories) == 38
    assert len(dataset.annotations) == 686


def test_sample3(tests_dir):
    with open(tests_dir / 'sample_3/groundtruths_coco.json') as fp:
        dataset = pcoco_decoder.load_object_detection(fp)

    assert len(dataset.images) == 7
    assert len(dataset.categories) == 1
    assert len(dataset.annotations) == 15


def test_sample2_result(tests_dir):
    with open(tests_dir / 'sample_2/groundtruths_coco.json') as fp:
        gold_dataset = pcoco_decoder.load_object_detection(fp)
    with open(tests_dir / 'sample_2/detections_coco.json') as fp:
        pred_dataset = pcoco_decoder.load_object_detection_result(fp, gold_dataset)

    assert len(gold_dataset.images) == 85
    assert len(gold_dataset.categories) == 38
    assert len(gold_dataset.annotations) == 686

    assert len(pred_dataset.images) == 85
    assert len(pred_dataset.categories) == 38
    assert len(pred_dataset.annotations) == 494


def test_sample3_result(tests_dir):
    with open(tests_dir / 'sample_3/groundtruths_coco.json') as fp:
        gold_dataset = pcoco_decoder.load_object_detection(fp)
    with open(tests_dir / 'sample_3/detections_coco.json') as fp:
        pred_dataset = pcoco_decoder.load_object_detection_result(fp, gold_dataset)

    assert len(gold_dataset.images) == 7
    assert len(gold_dataset.categories) == 1
    assert len(gold_dataset.annotations) == 15

    assert len(pred_dataset.images) == 7
    assert len(pred_dataset.categories) == 1
    assert len(pred_dataset.annotations) == 24