import math
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def test_pycocotools(sample_dir):
    coco = COCO(str(sample_dir / 'groundtruths_coco.json'))
    cats = coco.loadCats(coco.getCatIds())
    assert len(cats) == 38

    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['person'])
    assert catIds[0] == 26

    annIds = coco.getAnnIds(catIds=catIds)
    assert len(annIds) == 7

    imgIds = coco.getImgIds(catIds=catIds)
    assert len(imgIds) == 6

    annIds = coco.getAnnIds(imgIds=imgIds[1], catIds=catIds, iscrowd=None)
    assert len(annIds) == 1


def test_cocoeval(sample_dir):
    coco_gld = COCO(str(sample_dir / 'groundtruths_coco.json'))
    coco_rst = coco_gld.loadRes(str(sample_dir / 'detections_coco.json'))

    cocoEval = COCOeval(coco_gld, coco_rst, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    assert math.isclose(cocoEval.stats[1], 0.31195, rel_tol=1e-2)
    assert math.isclose(cocoEval.stats[-1], 0.307, rel_tol=1e-2)
