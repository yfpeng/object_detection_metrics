from podm.utils import load_data, load_data_coco
import math


def test_load_data(tests_dir):
    dir = tests_dir / 'sample_2'
    gt_bb = load_data(dir / 'groundtruths.json')
    pd_bb = load_data(dir / 'detections.json')

    assert len(gt_bb) == 686
    assert gt_bb[-1].image_id == '2007_001416'
    assert gt_bb[-1].category_id == 'cup'
    assert gt_bb[-1].score == 1

    assert len(pd_bb) == 494
    assert pd_bb[-1].image_id == '2007_001416'
    assert pd_bb[-1].category_id == 'chair'
    assert pd_bb[-1].score == 0.644847
    assert math.isclose(pd_bb[-1].score, 0.644847, rel_tol=1e-3)


def test_load_data_coco(tests_dir):
    dir = tests_dir / 'sample_2'
    gt_bb, pd_bb = load_data_coco(dir / 'groundtruths_coco.json',
                                                        dir / 'detections_coco.json')

    assert len(gt_bb) == 686
    assert gt_bb[-1].image_id == '2007_001416.txt.jpg'
    assert gt_bb[-1].category_id == 'cup'
    assert gt_bb[-1].score is None

    assert len(pd_bb) == 494
    assert pd_bb[-1].image_id == '2007_001416.txt.jpg'
    assert pd_bb[-1].category_id == 'chair'
    assert pd_bb[-1].score == 0.644847
    assert math.isclose(pd_bb[-1].score, 0.644847, rel_tol=1e-3)
