from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np


RESULT0_3 = {
    'person': {
        'ap': 0.245687
    }
}


def test_sample2_coco():
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
    print(cocoEval.stats[1])


if __name__ == '__main__':
    test_sample2_coco()
