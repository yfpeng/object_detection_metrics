from podm.pascal2coco import PascalVoc2COCO


def test_pascal2coco(sample_dir):
    converter = PascalVoc2COCO()
    dataset = converter.convert_gold(sample_dir / 'groundtruths.zip')
    assert len(dataset.images) == 85

    gold_dataset, pred_dataset = converter.convert_gold_pred(sample_dir / 'groundtruths.zip',
                                                             sample_dir / 'detections.zip')
    assert len(gold_dataset.images) == 85
    assert len(gold_dataset.annotations) == 686

    assert len(pred_dataset.images) == 85
    assert len(pred_dataset.annotations) == 494 - 44
