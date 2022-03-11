import pytest

from podm import coco, coco_decoder


def test_cat():
    dataset = coco.PCOCOBoundingBoxDataset()

    for i in range(0, 10):
        cat = coco.PCOCOCategory()
        cat.id = i
        cat.name = str(i)
        dataset.add_category(cat)
        assert dataset.get_category_id(cat.id).name == cat.name
        assert dataset.get_category_name(cat.name).name == cat.name
    assert dataset.get_max_category_id() == 9

    with pytest.raises(KeyError):
        cat = coco.PCOCOCategory()
        cat.id = 0
        dataset.add_category(cat)


def test_image():
    dataset = coco.PCOCOBoundingBoxDataset()

    for i in range(0, 10):
        img = coco.PCOCOImage()
        img.id = i
        img.file_name = str(i)
        dataset.add_image(img)
        assert dataset.get_image_id(img.id).file_name == img.file_name
        assert dataset.get_image_name(img.file_name).file_name == img.file_name

    with pytest.raises(KeyError):
        img = coco.PCOCOImage()
        img.id = 0
        dataset.add_image(img)


def test_gets(sample_dir):
    with open(sample_dir / 'groundtruths_coco.json') as fp:
        gold_dataset = coco_decoder.load_true_bounding_box_dataset(fp)

    # get all images containing given categories, select one at random
    cat_ids = gold_dataset.get_category_ids(category_names=['person'])
    assert 26 in cat_ids

    ann_ids = gold_dataset.get_annotation_ids(category_ids=cat_ids)
    assert len(ann_ids) == 7

    img_ids = gold_dataset.get_image_ids(category_ids=cat_ids)
    assert len(img_ids) == 6

    print(img_ids[1])

    ann_ids = gold_dataset.get_annotation_ids(image_ids=[img_ids[1]], category_ids=cat_ids)
    assert len(ann_ids) == 1
