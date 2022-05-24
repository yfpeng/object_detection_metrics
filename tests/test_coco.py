import pytest

from podm import coco, coco_decoder
from podm.box import Box


@pytest.fixture
def dataset():
    dataset = coco.PCOCOObjectDetectionDataset()

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


def test_cat(dataset):
    for i in range(0, 10):
        assert dataset.get_category(id=i).name == str(i)
        assert dataset.get_category(name=str(i)).id == i

    assert dataset.get_category(id=-1) is None
    assert dataset.get_category(name="-1") is None

    assert dataset.get_max_category_id() == 9

    cat_ids = dataset.get_category_ids(category_names=['1', '2', '-1'])
    assert 1 in cat_ids
    assert 2 in cat_ids
    assert 3 not in cat_ids

    cat_ids = dataset.get_category_ids(supercategory_names=['s1', 's2', '3'])
    assert 1 in cat_ids
    assert 2 in cat_ids
    assert 3 not in cat_ids

    cats = dataset.get_categories([1, 2])
    assert sorted(list(cat.id for cat in cats)) == [1, 2]

    with pytest.raises(KeyError):
        cat = coco.PCOCOCategory()
        cat.id = 0
        dataset.add_category(cat)

    with pytest.raises(KeyError):
        dataset.get_category()

    with pytest.raises(KeyError):
        dataset.get_category(id=0, name='0')


def test_ann(dataset):
    for i in range(0, 10):
        assert dataset.get_annotation(id=i).id == i

    assert dataset.get_annotation(id=-1) is None

    ann_ids = dataset.get_annotation_ids(image_ids=[1, -1])
    assert 1 in ann_ids
    assert 2 not in ann_ids
    assert -1 not in ann_ids

    ann_ids = dataset.get_annotation_ids(area_range=(100, 100))
    assert len(ann_ids) == 10

    ann_ids = dataset.get_annotation_ids(area_range=(99, 99))
    assert len(ann_ids) == 0

    ann_ids = dataset.get_annotation_ids(area_range=(0, 110))
    assert len(ann_ids) == 10

    ann_ids = dataset.get_annotation_ids(category_ids=[1, 2, -1])
    assert 1 in ann_ids
    assert 2 in ann_ids
    assert -1 not in ann_ids

    ann_ids = dataset.get_annotation_ids(image_ids=[1], category_ids=[1, 2])
    assert 1 in ann_ids
    assert 2 not in ann_ids

    anns = dataset.get_annotations([1, 2])
    assert sorted(list(ann.id for ann in anns)) == [1, 2]

    with pytest.raises(KeyError):
        ann = coco.PCOCOBoundingBox()
        ann.id = 0
        dataset.add_annotation(ann)


def test_image(dataset):
    for i in range(0, 10):
        assert dataset.get_image(id=i).file_name == str(i)
        assert dataset.get_image(file_name=str(i)).id == i

    assert dataset.get_image(id=-1) is None
    assert dataset.get_image(file_name='-1') is None

    img_ids = {img.id for img in dataset.get_images([0, 1, -2])}
    assert 0 in img_ids
    assert 1 in img_ids
    assert -2 not in img_ids

    img_ids = dataset.get_image_ids(category_ids=[1])
    assert 1 in img_ids

    img_ids = dataset.get_image_ids(category_ids=[1, 2])
    assert len(img_ids) == 0

    with pytest.raises(KeyError):
        img = coco.PCOCOImage()
        img.id = 0
        dataset.add_image(img)

    with pytest.raises(KeyError):
        dataset.get_image()

    with pytest.raises(KeyError):
        dataset.get_image(id=0, file_name='0')


def test_gets(dataset):
    # get all images containing given categories, select one at random
    cat_ids = dataset.get_category_ids(category_names=['1', '2'])
    assert len(cat_ids) == 2
    assert 1 in cat_ids
    assert 2 in cat_ids

    ann_ids = dataset.get_annotation_ids(category_ids=cat_ids)
    assert len(ann_ids) == 2
    assert 1 in ann_ids
    assert 2 in ann_ids


def test_segments():
    segments = coco.PCOCOSegments()
    segments.add_box(Box.of_box(0, 0, 10, 10))
    segments.add_box(Box.of_box(1, 1, 11, 11))
    assert segments.bbox == Box.of_box(0, 0, 11, 11)

    segments.add_segmentation(Box.of_box(2, 2, 12, 12).segment)
    assert segments.bbox == Box.of_box(0, 0, 12, 12)

    segments = coco.PCOCOSegments()
    assert segments.bbox is None
