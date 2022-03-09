import io

import pytest

from podm import coco_decoder, coco_encoder


@pytest.fixture
def coco_gld(sample_dir):
    with open(sample_dir / 'groundtruths_coco.json') as fp:
        coco_gld = coco_decoder.load_true_bounding_box_dataset(fp)
    return coco_gld


def test_load(coco_gld, tmp_path):
    tmp_file = tmp_path / 'foo.json'
    with open(tmp_file, 'w') as fp:
        coco_encoder.dump(coco_gld, fp)
    with open(tmp_file) as fp:
        coco_gld = coco_decoder.load_true_bounding_box_dataset(fp)

    assert len(coco_gld.images) == 85
    assert len(coco_gld.categories) == 38
    assert len(coco_gld.annotations) == 686


def test_loads(coco_gld):
    s = coco_encoder.dumps(coco_gld)
    coco_gld = coco_decoder.load_true_bounding_box_dataset(io.StringIO(s))

    assert len(coco_gld.images) == 85
    assert len(coco_gld.categories) == 38
    assert len(coco_gld.annotations) == 686

