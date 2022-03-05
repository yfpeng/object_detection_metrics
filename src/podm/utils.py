import json
from typing import List

from podm import BoundingBox


def load_data(pathname) -> List[BoundingBox]:
    with open(pathname, 'r') as fp:
        objs = json.load(fp)

    boxes = []
    for fig in objs:
        for box in fig['boxes']:
            if 'score' in box:
                score = box['score']
            else:
                score = 1
            bb = BoundingBox(
                fig['name'],
                box['label'],
                box['xtl'],
                box['ytl'],
                box['xbr'],
                box['ybr'],
                score)
            boxes.append(bb)
    return boxes
