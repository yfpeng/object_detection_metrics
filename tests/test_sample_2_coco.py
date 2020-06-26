import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from podm.podm import get_pascal_voc_metrics, MetricPerClass
from tests.utils import load_data, assert_results


def test_sample2_coco():
    dir = Path('tests/sample_2')
    coco_gld = COCO(dir / 'groundtruths_coco.json')
    coco_rst = coco_gld.loadRes(str(dir / 'detections_coco.json'))

    cocoEval = COCOeval(coco_gld, coco_rst, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print(cocoEval.stats[1])


if __name__ == '__main__':
    test_sample2_coco()