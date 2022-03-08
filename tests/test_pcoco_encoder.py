from podm import pcoco_decoder, pcoco_encoder


def test_sample2(tests_dir, tmp_path):
    with open(tests_dir / 'sample_2/groundtruths_coco.json') as fp:
        coco_gld = pcoco_decoder.load_true_bounding_box_dataset(fp)

    tmp_file = tmp_path / 'foo.json'
    with open(tmp_file, 'w') as fp:
        pcoco_encoder.dump(coco_gld, fp)

    with open(tmp_file) as fp:
        coco_gld = pcoco_decoder.load_true_bounding_box_dataset(fp)

    assert len(coco_gld.images) == 85
    assert len(coco_gld.categories) == 38
    assert len(coco_gld.annotations) == 686


def test_sample3(tests_dir, tmp_path):
    with open(tests_dir / 'sample_3/groundtruths_coco.json') as fp:
        coco_gld = pcoco_decoder.load_true_bounding_box_dataset(fp)

    tmp_file = tmp_path / 'foo.json'
    with open(tmp_file, 'w') as fp:
        pcoco_encoder.dump(coco_gld, fp)

    with open(tmp_file) as fp:
        coco_gld = pcoco_decoder.load_true_bounding_box_dataset(fp)

    assert len(coco_gld.images) == 7
    assert len(coco_gld.categories) == 1
    assert len(coco_gld.annotations) == 15
