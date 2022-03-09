import pytest

from podm import coco


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