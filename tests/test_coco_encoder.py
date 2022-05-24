import io

import pytest

from podm import coco_decoder, coco_encoder, coco
from podm.box import Box


@pytest.fixture
def dataset():
    dataset = coco.PCOCOObjectDetectionDataset()
    for i in range(0, 2):
        lic = coco.PCOCOLicense()
        lic.id = i
        lic.name = str(i)
        dataset.add_license(lic)

    for i in range(0, 10):
        cat = coco.PCOCOCategory()
        cat.id = i
        cat.name = str(i)
        cat.supercategory = 's%d' % i
        dataset.add_category(cat)

    for i in range(0, 10):
        img = coco.PCOCOImage()
        img.id = i
        img.file_name = str(i)
        dataset.add_image(img)

    for i in range(0, 10):
        ann = coco.PCOCOBoundingBox()
        ann.id = i
        ann.image_id = i
        ann.category_id = i
        ann.set_box(Box.of_box(0, 0, 10, 10))
        dataset.add_annotation(ann)

    return dataset


def test_load(dataset, tmp_path):
    tmp_file = tmp_path / 'foo.json'
    with open(tmp_file, 'w') as fp:
        coco_encoder.dump(dataset, fp)
    with open(tmp_file) as fp:
        dataset = coco_decoder.load_true_object_detection_dataset(fp)

    assert len(dataset.images) == 10
    assert len(dataset.categories) == 10
    assert len(dataset.annotations) == 10
    assert len(dataset.licenses) == 2


def test_loads(dataset):
    s = coco_encoder.dumps(dataset)
    dataset = coco_decoder.load_true_object_detection_dataset(io.StringIO(s))

    assert len(dataset.images) == 10
    assert len(dataset.categories) == 10
    assert len(dataset.annotations) == 10
    assert len(dataset.licenses) == 2

