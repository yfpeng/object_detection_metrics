from podm import pcoco_decoder, pcoco_encoder


def test_sample2(tests_dir, tmp_path):
    with open(tests_dir / 'sample_2/groundtruths_coco.json') as fp:
        coco_gld = pcoco_decoder.load_object_detection(fp)

    s = pcoco_encoder.dumps(coco_gld)
    coco_gld = pcoco_decoder.loads_object_detection(s)

    assert len(coco_gld.images) == 85
    assert len(coco_gld.categories) == 38
    assert len(coco_gld.annotations) == 686


def test_sample3(tests_dir):
    with open(tests_dir / 'sample_3/groundtruths_coco.json') as fp:
        coco_gld = pcoco_decoder.load_object_detection(fp)

    s = pcoco_encoder.dumps(coco_gld)
    coco_gld = pcoco_decoder.loads_object_detection(s)

    assert len(coco_gld.images) == 7
    assert len(coco_gld.categories) == 1
    assert len(coco_gld.annotations) == 15
