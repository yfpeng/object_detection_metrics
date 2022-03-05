import math
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def test_sample2():
    dir = Path('tests/sample_2')
    coco_gld = COCO(dir / 'groundtruths_coco.json')
    coco_rst = coco_gld.loadRes(str(dir / 'detections_coco.json'))

    cocoEval = COCOeval(coco_gld, coco_rst, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    assert math.isclose(cocoEval.stats[1], 0.31195, rel_tol=1e-2)
    assert math.isclose(cocoEval.stats[-1], 0.307, rel_tol=1e-2)


def test_sample3():
    dir = Path('tests/sample_3')
    coco_gld = COCO(dir / 'groundtruths_coco.json')
    coco_rst = coco_gld.loadRes(str(dir / 'detections_coco.json'))

    cocoEval = COCOeval(coco_gld, coco_rst, iouType='bbox')
    cocoEval.params.catIds = coco_gld.getCatIds(catNms=['person'])
    # cocoEval.params.iouThrs = np.linspace(.5, 1, int(np.round((1 - .5) / .05)) + 1, endpoint=True)
    # cocoEval.params.areaRngLbl = ['all']
    # cocoEval.params.maxDets = [1, 10, 200]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    assert math.isclose(cocoEval.stats[1], 0.0231, rel_tol=1e-2)
    assert math.isclose(cocoEval.stats[-1], -1.000, rel_tol=1e-2)


if __name__ == '__main__':
    test_sample3()